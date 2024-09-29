from collections import namedtuple, deque
import pickle
from typing import List
import events as e
import numpy as np

# Import feature extraction functions from callbacks
from .callbacks import extract_features#, enters_bomb_range_safe, walks_into_explosion
from .callbacks import include_agents_in_field
from .callbacks import POSSIBLE_ACTIONS, NUM_FEATURES
from itertools import compress 

import matplotlib.pyplot as plt


# Constants for action and state history
PAST_ACTIONS_LENGTH = 4
PAST_STATES_LENGTH = 2
MAX_ROUNDS = 1000

# Define custom events for training

CONTINUED_LOOP = "CONTINUED_LOOP"
APPROACHED_COIN = "APPROACHED_COIN"
PLACED_BOMB_NEAR_CRATE = "PLACED_BOMB_NEAR_CRATE"
APPROACHED_CRATE = "APPROACHED_CRATE"
MOVED_FROM_BOMB = "MOVED_FROM_BOMB"
EXITED_BOMB_RANGE = "EXITED_BOMB_RANGE"
ENTERED_CRATE_TRAP = "ENTERED_CRATE_TRAP"
EXPLOSION_HIT = "EXPLOSION_HIT"
APPROACHED_BOMB = "APPROACHED_BOMB"
UNNECESSARY_BOMB = "UNNECESSARY_BOMB"
SAFE_START = "SAFE_START"
BOMB_NEAR_AGENT = "BOMB_NEAR_AGENT"
COLLECTED_NEAREST_COIN = "COLLECTED_NEAREST_COIN"
ACTION_WAIT = "ACTION_WAIT"

# Initialize training parameters
def setup_training(self):
    self.exp_decay = 0.99
    self.reward_array = []
    
    """
    Sets up the training parameters for the agent, including learning rate and discount factor.
    """
    self.learning_rate = 0.18 
    self.discount_factor = 0.92 
    self.epsilon_decay = 1
    self.logger.info("Training setup complete.")
    self.coins_round = 0
    self.crates_round = 0
    

# Handle game events and update model
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Processes game events, updates the agent's model based on rewards, and logs relevant information.
    """
    
    # Initialize training if necessary
    if old_game_state is not None:
        if old_game_state["step"] == 1:
            setup_training(self)

    # Special handling for the first step of the game
    if new_game_state["step"] == 1:
        old_game_state = new_game_state
        self.past_actions.appendleft(0) 
        feature_values = extract_features(old_game_state, self_action, self)
        self.past_actions.append(self.selected_action) 

        # Check for safe start move
        if feature_values[10] != 0:
            events.append(SAFE_START)

        # Calculate reward and Q-value for the new state
        reward = calculate_reward(self, events)
        weights = self.agent_model
        action_features = [extract_features(new_game_state, a, self) for a in POSSIBLE_ACTIONS]
        best_action_index = np.argmax([(weights * features).sum() for features in action_features])
        best_action_features = extract_features(new_game_state, POSSIBLE_ACTIONS[best_action_index], self)
        max_q_new_state = sum(weights * best_action_features)

        # Update model weights
        self.past_actions.appendleft(0) 
        features = feature_values
        self.past_actions.append(self.selected_action) 
        for i in range(NUM_FEATURES):
            weights[i] += self.learning_rate * features[i] * (reward + self.discount_factor * max_q_new_state - sum(weights * features))

        # Update model and epsilon
        self.agent_model = weights
        self.logger.info(f'Exploration rate: {self.exploration_rate}')
        # self.exploration_rate *= self.epsilon_decay
        self.past_states.append(new_game_state)
        old_game_state = None 

    # Process events for non-initial game steps
    if extract_features(old_game_state, self_action, self) is not None:
        include_agents_in_field(old_game_state)
        include_agents_in_field(new_game_state)

        self.past_actions.appendleft(0) 
        feature_values = extract_features(old_game_state, self_action, self)
        self.past_actions.append(self.selected_action) 

        # Assign events based on feature values
        if feature_values[1] != 0:
            events.append(CONTINUED_LOOP)
            is_looping = 1
            self.is_action_loop = 1
        else:
            is_looping = 0
            self.is_action_loop = 0
            
        if feature_values[2] != 0:
            events.append(APPROACHED_COIN)
        if feature_values[3] != 0:
            events.append(PLACED_BOMB_NEAR_CRATE)
        if feature_values[4] != 0:
            events.append(APPROACHED_CRATE)
        if feature_values[5] != 0:
            events.append(MOVED_FROM_BOMB)
        if feature_values[6] != 0:
            events.append(EXITED_BOMB_RANGE)
        if feature_values[7] != 0:
            events.append(ENTERED_CRATE_TRAP)
        if feature_values[9] != 0:
            events.append(APPROACHED_BOMB)
        if feature_values[10] != 0:
            events.append(UNNECESSARY_BOMB)
        if feature_values[12] != 0:
            events.append(BOMB_NEAR_AGENT)
        if feature_values[13] != 0:
            events.append(COLLECTED_NEAREST_COIN)
        if feature_values[-1] != 0:
            events.append(ACTION_WAIT)

        # Log agent position and exploration rate
        agent_x, agent_y = old_game_state['self'][3]
        self.logger.info(f'')
        self.logger.info(f'Exploration rate: {self.exploration_rate}')
        self.logger.info(f'Current position:   x: {agent_x}  y: {agent_y}')

        # Calculate reward and Q-value for the new state
        reward = calculate_reward(self, events)
        weights = self.agent_model
        action_features = [extract_features(new_game_state, a, self) for a in POSSIBLE_ACTIONS]
        best_action_index = np.argmax([(weights * features).sum() for features in action_features])
        best_action_features = extract_features(new_game_state, POSSIBLE_ACTIONS[best_action_index], self)
        max_q_new_state = sum(weights * best_action_features)

        # Update model weights
        self.past_actions.appendleft(0) 
        features = feature_values
        self.past_actions.append(self.selected_action) 
        for i in range(NUM_FEATURES):
            weights[i] += self.learning_rate * features[i] * (reward + self.discount_factor * max_q_new_state - sum(weights * features))

        # Update model and epsilon
        self.agent_model = weights
        # self.exploration_rate *= self.epsilon_decay
        self.past_states.append(new_game_state)

        self.reward_array.append(reward)
        if "COIN_COLLECTED" in events:
            self.coins_round += 1
        if "CRATE_DESTROYED" in events:
             self.crates_round += 1
        # print(is_looping, self.is_action_loop)



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Processes events at the end of a round, updates the model, and logs performance metrics.
    """
    # print(self.exploration_rate)
    if last_game_state['round'] == 1:
        self.reward_array_rounds = []
        self.coins_array = []
        self.crates_array = []
        
    self.coins_array.append(self.coins_round)
    self.coins_round = 0
    self.crates_array.append(self.crates_round)
    self.crates_round = 0
    self.reward_array_rounds.append(np.mean(self.reward_array))
    
    if last_game_state['round'] == MAX_ROUNDS:
        plt.plot(self.coins_array)
        plt.title("Number of collected coins per Round")
        plt.xlabel("Rounds")
        plt.ylabel("collected Coins")
        plt.savefig("coins.pdf", dpi = 300)
    
    # if last_game_state['round'] == MAX_ROUNDS:
    #     plt.plot(self.crates_array)
    #     plt.title("Number of destroyed crates per Round")
    #     plt.xlabel("Round")
    #     plt.ylabel("destroyed Crates")
    #     plt.savefig("crates.pdf", dpi = 300)
        
    # if last_game_state['round'] == MAX_ROUNDS:
    #     plt.plot(self.reward_array_rounds)
    #     plt.title("average Reward per Round")
    #     plt.xlabel("Rounds")
    #     plt.ylabel("Reward")
    #     plt.savefig("reward.pdf", dpi = 300)

    self.logger.info("Final Round Evaluation")
    old_game_state = last_game_state
    self_action = last_action

    # Include agent positions in the game field
    include_agents_in_field(old_game_state)

    # Determine if the agent is in an action loop
    if self.is_action_loop != 0:
        is_looping = 1
    else:
        is_looping = 0
    self.logger.info(f"Looping status: {is_looping}")

    # Process events for the last game state
    if extract_features(old_game_state, self_action, self) is not None:
        self.past_actions.appendleft(0) 
        feature_values = extract_features(old_game_state, self_action, self)
        self.past_actions.append(self.selected_action) 

        # Assign events based on feature values
        if is_looping != 0:
            events.append(CONTINUED_LOOP)
        if feature_values[2] != 0:
            events.append(APPROACHED_COIN)
        if feature_values[3] != 0:
            events.append(PLACED_BOMB_NEAR_CRATE)
        if feature_values[4] != 0:
            events.append(APPROACHED_CRATE)
        if feature_values[5] != 0:
            events.append(MOVED_FROM_BOMB)
        if feature_values[6] != 0:
            events.append(EXITED_BOMB_RANGE)
        if feature_values[7] != 0:
            events.append(ENTERED_CRATE_TRAP)
        if feature_values[8] != 0:
            events.append(EXPLOSION_HIT)
        if feature_values[9] != 0:
            events.append(APPROACHED_BOMB)
        if feature_values[10] != 0:
            events.append(UNNECESSARY_BOMB)
        if feature_values[12] != 0:
            events.append(BOMB_NEAR_AGENT)
        if feature_values[13] != 0:
            events.append(COLLECTED_NEAREST_COIN)
        if feature_values[-1] != 0:
            events.append(ACTION_WAIT)

        # Log final agent information
        agent_x, agent_y = old_game_state['self'][3]
        self.logger.info(f'')
        self.logger.info(f'Exploration rate: {self.exploration_rate}')
        self.logger.info(f'Final position:   x: {agent_x}  y: {agent_y}')

        # Calculate reward and update model
        reward = calculate_reward(self, events)
        weights = self.agent_model
        self.past_actions.appendleft(0) 
        features = feature_values
        self.past_actions.append(self.selected_action) 
        for i in range(NUM_FEATURES):
            weights[i] += self.learning_rate * features[i] * (reward - sum(weights * features))
        self.agent_model = weights

    self.logger.info(f"END OF GAME ---------------- Step: {last_game_state['step']} ")

    
    
    # Save the trained model
    with open("saved_model.pt", "wb") as file:
        pickle.dump(self.agent_model, file)

# Calculate reward based on game events
def calculate_reward(self, events: List[str]) -> int:
    """
    Assigns rewards based on the events that occurred during the game.
    """

    event_rewards = {
        e.MOVED_LEFT: 8,
        e.MOVED_RIGHT: 8,
        e.MOVED_UP: 8,
        e.MOVED_DOWN: 8,
        e.WAITED: -5,
        e.INVALID_ACTION: -120,
        e.BOMB_DROPPED: 0,
        e.CRATE_DESTROYED: 90,
        e.COIN_FOUND: 30,
        e.COIN_COLLECTED: 30,
        CONTINUED_LOOP: -15,
        APPROACHED_COIN: 10,
        PLACED_BOMB_NEAR_CRATE: 50,
        APPROACHED_CRATE: 30,  
        MOVED_FROM_BOMB: 20, 
        EXITED_BOMB_RANGE: 50,  
        ENTERED_CRATE_TRAP: -45,
        EXPLOSION_HIT: -120,  
        APPROACHED_BOMB: -36,  
        UNNECESSARY_BOMB: -60,  
        SAFE_START: 120,  
        BOMB_NEAR_AGENT: 20,  
        COLLECTED_NEAREST_COIN: 50, 
        ACTION_WAIT: -5, 
    }
    total_reward = 0
    for event in events:
        if event in event_rewards:
            total_reward += event_rewards[event]

    # Logging information about rewards and actions
    if self.action_type == 1:
        self.logger.info(f"RANDOM ACTION: {self.past_actions[-1]}")
    if self.action_type == 2:
        self.logger.info(f"CHOSEN ACTION: {self.past_actions[-1]}")

    self.logger.info(f"Reward: {total_reward} for events: {', '.join(events)}")


    return total_reward
