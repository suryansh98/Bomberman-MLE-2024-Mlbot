import os
import pickle
import random

import numpy as np
from collections import deque
from itertools import compress 

# Define possible actions and game features
POSSIBLE_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']#, 'WAIT', 'BOMB']
NUM_FEATURES = 15
PAST_ACTIONS_LENGTH = 4
PAST_STATES_LENGTH = 2

def setup(self):
    """
    Sets up the agent's model, either loading from a file or creating a new one.
    Also initializes variables for tracking actions, states, and performance metrics.
    """

    # Load existing model or create a new one
    if not os.path.isfile("saved_model.pt"):
        self.logger.info("Creating new model")
        initial_weights = np.random.rand(NUM_FEATURES)
        self.agent_model = initial_weights

        # Initialize epsilon only when creating a new model
        self.exploration_rate = 0.9  # Initial exploration rate 
        print("NOW")
    else:
        self.logger.info("Loading model from file")
        with open("saved_model.pt", "rb") as model_file:
            self.agent_model = pickle.load(model_file)

    # Initialize these variables for EVERY round
    self.past_actions = deque(maxlen=PAST_ACTIONS_LENGTH)
    self.past_states = deque(maxlen=PAST_STATES_LENGTH)
    self.is_action_loop = 0
    self.loop_counter = 0
    self.round_wins = []
    self.action_type = 0
    self.wall_crate_collisions = 0
    self.action_loop_count = 0
    self.explosion_collisions = 0
    self.selected_action = None
    self.exp_rate_min = 0.1
    self.exp_rate_decay= 0.0001


# Function to choose an action
def act(self, game_state: dict) -> str:
    """
    Selects an action for the agent based on the current game state.
    Uses an epsilon-greedy strategy to balance exploration and exploitation.
    """
    # print(self.agent_model)
    # Initialize agent if necessary
    if game_state is not None:
        if game_state["step"] == 1:
            setup(self)
            
    
    if game_state['round'] == 1:
        self.exploration_rate = 1
    
    if self.exploration_rate > self.exp_rate_min:
        self.exploration_rate = self.exploration_rate - self.exp_rate_decay

    # Logging information
    self.logger.info(f'\n-------------------- Step: {game_state["step"]}')
    self.logger.info(f'Bomb available: {game_state["self"][2]}')

    # Include agent positions in the game field
    include_agents_in_field(game_state)

    # Exploration: Choose a random action
    if self.train and random.random() < self.exploration_rate:
        action = np.random.choice(POSSIBLE_ACTIONS, p=[1/4, 1/4, 1/4, 1/4])#, 1/6, 1/6]) 
        # action = np.random.choice(POSSIBLE_ACTIONS, p=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]) 
        self.action_type = 1


    # Exploitation: Choose the best action based on the model
    else:
        weights = self.agent_model
        # print('weights',weights)
        action_features = [extract_features(game_state, a, self) for a in POSSIBLE_ACTIONS]
        # print('action_features',action_features)
        action_scores = [(weights * features).sum() for features in action_features]
        best_action_index = np.argmax(action_scores)
        action = POSSIBLE_ACTIONS[best_action_index]        
        self.action_type = 2

    # Store the selected action
    self.selected_action = action

    # Logging the chosen action
    if len(self.past_actions) == 4:
        self.logger.info(f"Previous 4 actions: {self.past_actions}")
        self.logger.info(f"UP: {extract_features(game_state, POSSIBLE_ACTIONS[0], self)[2]}, RIGHT: {extract_features(game_state, POSSIBLE_ACTIONS[1], self)[2]}, DOWN: {extract_features(game_state, POSSIBLE_ACTIONS[2], self)[2]}, LEFT: {extract_features(game_state, POSSIBLE_ACTIONS[3], self)[2]}")

    if self.action_type == 2:
        self.logger.info(f"Chosen action: {self.selected_action}")
    elif self.action_type == 1:
        self.logger.info(f"Random action: {self.selected_action}")

    self.past_actions.append(self.selected_action) 

    # Check for action loops
    if len(self.past_actions) == 4:
        self.logger.info(f"Previous 4 actions (including current): {self.past_actions}")
        if (self.past_actions[0] == self.past_actions[2]) and (self.past_actions[1] == self.past_actions[3]) and (self.past_actions[0] != self.past_actions[1]) and (self.past_actions[2] != self.past_actions[3]):
            self.logger.info("Action loop detected!")

    # Update action loop variables
    self.past_actions.appendleft(0) 
    # extract_features(game_state, self.selected_action, self)
    self.logger.info(f"Is action loop: {self.is_action_loop}")
    self.past_actions.append(self.selected_action) 
    # print(self.exploration_rate)
    return action

def include_agents_in_field(game_state):
    """
    Updates the game field to include agent positions.
    Marks agents' locations with the value 2.
    """
    agent_locations = [item[3] for item in game_state["others"]]

    for x, y in agent_locations:
        game_state["field"][x][y] = 2

# Function to extract features from the game state
def extract_features(game_state: dict, action, self) -> np.array:
    """
    Extracts features from the game state and action. These features are used for
    training the agent's model and for making decisions.
    """

    if game_state is None:
        return None

    # Calculate various features 
    feature_list = [
        f0(),
        is_looping_action(game_state, action, self),
        moves_towards_nearest_coin(game_state, action, self),
        drops_bomb_near_crate(find_nearest_crate(game_state, action, self), game_state, action, self),
        moves_towards_nearest_crate(find_nearest_crate(game_state, action, self), game_state, action, self),
        moves_to_avoid_bomb(find_dangerous_bombs(game_state, action, self), game_state, action, self),
        exits_bomb_range(find_dangerous_bombs(game_state, action, self), game_state, action, self),
        enters_crate_trap(find_dangerous_bombs(game_state, action, self), game_state, action, self),
        walks_into_explosion(game_state, action, self),
        approaches_dangerous_bomb(find_dangerous_bombs(game_state, action, self), game_state, action, self),
        drops_bomb_unnecessarily(game_state, action, self),
        safe_start_move(game_state, action, self),
        drops_bomb_near_agent(find_nearest_agent(game_state, action, self), game_state, action, self),
        collects_nearest_coin(game_state, action, self),
        waits(game_state, action, self),
    ]
    return np.array(feature_list) 

#%%
# Feature functions

def f0():
    return 1

def enters_crate_trap(dangerous_bombs_and_number_of_dangerous_bombs, game_state, action, self):
    dangerous_bombs = [bomb[0] for bomb in dangerous_bombs_and_number_of_dangerous_bombs[0]] 
    number_of_dangerous_bombs = dangerous_bombs_and_number_of_dangerous_bombs[1]

    agent_x, agent_y = game_state['self'][3]

    # Case 1: Only one dangerous bomb
    if number_of_dangerous_bombs == 1:
        bomb_x, bomb_y = next(bomb for bomb in dangerous_bombs if bomb is not None)

        # Check movement in each direction
        if action == 'UP':
            if abs(bomb_y - (agent_y - 1)) > abs(bomb_y - agent_y) and game_state['field'][agent_x][agent_y - 1] == 0:
                if (agent_y - 1) % 2 == 0 and (agent_y - 1) != 0 and game_state['field'][agent_x][agent_y - 2] != 0:
                    return 1
                return 0

        if action == 'RIGHT':
            if abs(bomb_x - (agent_x + 1)) > abs(bomb_x - agent_x) and game_state['field'][agent_x + 1][agent_y] == 0:
                if (agent_x + 1) % 2 == 0 and (agent_x + 1) != len(game_state['field']) - 1 and game_state['field'][agent_x + 2][agent_y] != 0:
                    return 1
                return 0

        if action == 'DOWN':
            if abs(bomb_y - (agent_y + 1)) > abs(bomb_y - agent_y) and game_state['field'][agent_x][agent_y + 1] == 0:
                if (agent_y + 1) % 2 == 0 and (agent_y + 1) != len(game_state['field'][0]) - 1 and game_state['field'][agent_x][agent_y + 2] != 0:
                    return 1
                return 0

        if action == 'LEFT':
            if abs(bomb_x - (agent_x - 1)) > abs(bomb_x - agent_x) and game_state['field'][agent_x - 1][agent_y] == 0:
                if (agent_x - 1) % 2 == 0 and (agent_x - 1) != 0 and game_state['field'][agent_x - 2][agent_y] != 0:
                    return 1
                return 0

    return 0

def is_looping_action(game_state, action, self):
    """
    Checks if the agent is repeating a sequence of two alternating actions.
    """
    if len(self.past_actions) == 4:
        first_action, second_action = self.past_actions[1], self.past_actions[2]
        third_action, fourth_action = self.past_actions[3], action

        if first_action == third_action and second_action == fourth_action and first_action != second_action:
            self.is_action_loop = 1
            return -1

    self.is_action_loop = 0
    return 0


def moves_to_avoid_bomb(dangerous_bombs_and_number_of_dangerous_bombs, game_state, action, self):

    dangerous_bombs = [item[0] for item in dangerous_bombs_and_number_of_dangerous_bombs[0]] 
    number_of_dangerous_bombs = dangerous_bombs_and_number_of_dangerous_bombs[1]

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]


    #case 1:  one bomb-------------------------------------------------------
    if number_of_dangerous_bombs == 1:
        
        #get the bomb which is not 'None'
        bomb_coord = dangerous_bombs[ [i for i in range(len(dangerous_bombs)) if dangerous_bombs[i] != None][0] ]

        bomb_coord_x = bomb_coord[0]
        bomb_coord_y = bomb_coord[1]

        x, y = bomb_coord_x, bomb_coord_y 


        # the next direction to be farther from the bomb
        if action == 'UP':
            if abs(y - (agent_coord_y - 1)) > abs(y - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                #avoid crate trap
                """
                ____W_
                _B__xC  (moved right in this case)
                ____W_
                """
                #if agent moved to free lane, crate problem cant occure
                if ((agent_coord_y - 1) % 2) != 0:
                    return 1
                #if moved to |_|_|x|_|_| lane, check if agent would be trapped by crate above
                elif ((agent_coord_y - 1) % 2) == 0 and ((agent_coord_y - 1) != 0) :
                    if game_state['field'][agent_coord_x][agent_coord_y-2] != 0:
                        return 0

                return 1

        if action == 'RIGHT':
            if abs(x - (agent_coord_x + 1)) > abs(x - agent_coord_x) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                if ((agent_coord_x + 1) % 2) != 0:
                    return 1
                elif ((agent_coord_x + 1) % 2) == 0 and ((agent_coord_x + 1) != len(game_state['field'])-1):
                    if game_state['field'][agent_coord_x+2][agent_coord_y] != 0:
                        return 0
                        
                return 1

        if action == 'DOWN':
            if abs(y - (agent_coord_y + 1)) > abs(y - agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                if ((agent_coord_y + 1) % 2) != 0:
                    return 1
                elif ((agent_coord_y + 1) % 2) == 0 and ((agent_coord_y + 1) != len(game_state['field'][:][0])-1) :
                    if game_state['field'][agent_coord_x][agent_coord_y+2] != 0:
                        return 0
                        
                return 1

        if action == 'LEFT':
            if abs(x - (agent_coord_x - 1)) > abs(x - agent_coord_x) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                if ((agent_coord_x - 1) % 2) != 0:
                    return 1
                elif ((agent_coord_x - 1) % 2) == 0 and ((agent_coord_x - 1) != 0):
                    if game_state['field'][agent_coord_x-2][agent_coord_y] != 0:
                        return 0
                        
                return 1
    return 0



def moves_towards_nearest_coin(game_state, action, self):
    """
    Checks if the action moves the agent closer to the nearest coin, 
    avoiding walls and crates, and preventing up-down loops.
    """
    
    agent_x, agent_y = game_state['self'][3]

    # Find the closest coin
    coin_locations = game_state['coins']
    closest_coin = min(coin_locations, key=lambda c: np.linalg.norm([c[0] - agent_x, c[1] - agent_y]), default=None)

    if closest_coin is None:
        return 0

    x, y = closest_coin
    move_map = {
        'UP': (0, -1),
        'RIGHT': (1, 0),
        'DOWN': (0, 1),
        'LEFT': (-1, 0)
    }

    if action in move_map:
        move_x, move_y = move_map[action]
        new_x, new_y = agent_x + move_x, agent_y + move_y

        # Check if the move brings the agent closer and there are no walls or crates
        if abs(x - new_x) < abs(x - agent_x) or abs(y - new_y) < abs(y - agent_y):
            if game_state['field'][new_x][new_y] == 0:  # No wall or crate
                # Avoid loop: ensure we aren't just alternating in a fixed position
                if (new_x % 2 != 0 or new_y % 2 != 0) or (abs(x - new_x) != 0 or abs(y - new_y) != 0):
                    return 1
                # If already aligned with the coin in one axis
                if abs(x - agent_x) == 0 or abs(y - agent_y) == 0:
                    return 1

    return 0

def drops_bomb_near_crate(closest_crate, game_state, action, self):
    """
    Determines if the agent should drop a bomb near the closest crate.
    """
    agent_coord_x, agent_coord_y = game_state['self'][3]

    if closest_crate is not None:
        crate_x, crate_y = closest_crate

        if action == 'BOMB' and game_state['self'][2]:  # Check if bomb is available
            # Check if crate is within bomb range in y direction
            if abs(crate_y - agent_coord_y) <= 3 and crate_x == agent_coord_x:
                if agent_coord_x % 2 != 0:  # No wall in between
                    return 1

            # Check if crate is within bomb range in x direction
            if abs(crate_x - agent_coord_x) <= 3 and crate_y == agent_coord_y:
                if agent_coord_y % 2 != 0:  # No wall in between
                    return 1

    return 0

def find_nearest_crate(game_state, action, self):
    """
    Finds the nearest crate based on the agent's current coordinates.
    """

    agent_coord_x, agent_coord_y = game_state['self'][3]

    # Generate a list of crate locations (x, y) where the field equals 1 (crate)
    crate_locations = list(zip(*np.where(game_state['field'] == 1)))

    closest_crate = None
    closest_dist = float('inf')

    # Find the closest crate
    for crate_x, crate_y in crate_locations:
        dist = np.linalg.norm([crate_x - agent_coord_x, crate_y - agent_coord_y])
        if dist < closest_dist:
            closest_dist = dist
            closest_crate = (crate_x, crate_y)

    return closest_crate

def moves_towards_nearest_crate(target_crate, game_state, action, self):
    """
    Determines if the agent should move towards the target crate while avoiding walls or obstacles.
    """
    agent_x, agent_y = game_state['self'][3]

    if target_crate:
        crate_x, crate_y = target_crate

        # Helper function to validate moves
        def is_valid_move(new_x, new_y):
            return game_state['field'][new_x][new_y] == 0 and not (crate_x == new_x and crate_y == new_y)

        # Check each action and decide if moving towards the crate is valid
        if action == 'UP' and abs(crate_y - (agent_y - 1)) < abs(crate_y - agent_y) and is_valid_move(agent_x, agent_y - 1):
            return 1 if (agent_y - 1) % 2 != 0 or abs(crate_y - (agent_y - 1)) != 0 else 0

        if action == 'RIGHT' and abs(crate_x - (agent_x + 1)) < abs(crate_x - agent_x) and is_valid_move(agent_x + 1, agent_y):
            return 1 if (agent_x + 1) % 2 != 0 or abs(crate_x - (agent_x + 1)) != 0 else 0

        if action == 'DOWN' and abs(crate_y - (agent_y + 1)) < abs(crate_y - agent_y) and is_valid_move(agent_x, agent_y + 1):
            return 1 if (agent_y + 1) % 2 != 0 or abs(crate_y - (agent_y + 1)) != 0 else 0

        if action == 'LEFT' and abs(crate_x - (agent_x - 1)) < abs(crate_x - agent_x) and is_valid_move(agent_x - 1, agent_y):
            return 1 if (agent_x - 1) % 2 != 0 or abs(crate_x - (agent_x - 1)) != 0 else 0

    return 0

# def moves_towards_nearest_crate(target_crate, game_state, action, self):
#     """
#     Determines if the agent should move towards the target crate while avoiding walls or obstacles.
#     """
#     agent_x, agent_y = game_state['self'][3]

#     if target_crate is not None:
#         crate_x, crate_y = target_crate

#         # Check movement for each action
#         if action == 'UP':
#             if abs(crate_y - (agent_y - 1)) < abs(crate_y - agent_y) and game_state['field'][agent_x][agent_y - 1] == 0:
#                 if crate_y == agent_y - 1 and crate_x == agent_x:
#                     return 0  # Prevent moving into the crate
#                 if (agent_y - 1) % 2 != 0 or abs(crate_y - (agent_y - 1)) != 0:
#                     return 1

#         if action == 'RIGHT':
#             if abs(crate_x - (agent_x + 1)) < abs(crate_x - agent_x) and game_state['field'][agent_x + 1][agent_y] == 0:
#                 if crate_x == agent_x + 1 and crate_y == agent_y:
#                     return 0  # Prevent moving into the crate
#                 if (agent_x + 1) % 2 != 0 or abs(crate_x - (agent_x + 1)) != 0:
#                     return 1

#         if action == 'DOWN':
#             if abs(crate_y - (agent_y + 1)) < abs(crate_y - agent_y) and game_state['field'][agent_x][agent_y + 1] == 0:
#                 if crate_y == agent_y + 1 and crate_x == agent_x:
#                     return 0  # Prevent moving into the crate
#                 if (agent_y + 1) % 2 != 0 or abs(crate_y - (agent_y + 1)) != 0:
#                     return 1

#         if action == 'LEFT':
#             if abs(crate_x - (agent_x - 1)) < abs(crate_x - agent_x) and game_state['field'][agent_x - 1][agent_y] == 0:
#                 if crate_x == agent_x - 1 and crate_y == agent_y:
#                     return 0  # Prevent moving into the crate
#                 if (agent_x - 1) % 2 != 0 or abs(crate_x - (agent_x - 1)) != 0:
#                     return 1

#     return 0


def walks_into_explosion(game_state, action, self):
    """
    Checks if the agent's action would lead into an explosion.
    """
    agent_x, agent_y = game_state['self'][3]

    # Mapping actions to coordinate changes
    action_offsets = {
        'UP': (0, -1),
        'RIGHT': (1, 0),
        'DOWN': (0, 1),
        'LEFT': (-1, 0)
    }

    if action in action_offsets:
        offset_x, offset_y = action_offsets[action]
        if game_state['explosion_map'][agent_x + offset_x][agent_y + offset_y] != 0:
            return 1

    return 0


def safe_start_move(game_state, action, self):
    """
    Checks if the agent's action in the starting position could lead to danger.
    """
    agent_x, agent_y = game_state['self'][3]
    field_width = len(game_state['field'][0])
    field_height = len(game_state['field'])

    corners = {
        'upper_left': ((agent_x <= 2) and (agent_y <= 2), ["RIGHT", "DOWN"]),
        'upper_right': ((agent_x >= field_width - 3) and (agent_y <= 2), ["LEFT", "DOWN"]),
        'bottom_left': ((agent_x <= 2) and (agent_y >= field_height - 3), ["RIGHT", "UP"]),
        'bottom_right': ((agent_x >= field_width - 3) and (agent_y >= field_height - 3), ["LEFT", "UP"]),
    }

    if game_state["step"] == 1:
        for condition, actions in corners.values():
            if condition and action in actions:
                return 1

    return 0



def collects_nearest_coin(game_state, action, self):
    """
    Checks if the agent can collect the closest coin without hitting a wall or crate.
    """
    agent_x, agent_y = game_state['self'][3]
    coin_locations = game_state['coins']
    
    # Find the closest coin
    closest_coin = min(
        coin_locations,
        key=lambda coin: np.linalg.norm([coin[0] - agent_x, coin[1] - agent_y]),
        default=None
    )

    # If a closest coin is found, check the action
    if closest_coin:
        coin_x, coin_y = closest_coin
        
        # Define movement checks based on action
        moves = {
            'UP': (coin_x == agent_x and coin_y == agent_y - 1, (agent_x, agent_y - 1)),
            'RIGHT': (coin_x == agent_x + 1 and coin_y == agent_y, (agent_x + 1, agent_y)),
            'DOWN': (coin_x == agent_x and coin_y == agent_y + 1, (agent_x, agent_y + 1)),
            'LEFT': (coin_x == agent_x - 1 and coin_y == agent_y, (agent_x - 1, agent_y))
        }

        # Check if the action is valid and there's no wall or crate
        if action in moves:
            valid_move, position = moves[action]
            if valid_move and game_state['field'][position[0]][position[1]] == 0:
                return 1

    return 0



def waits(game_state, action, self):

    if action == 'WAIT':
        return 1

    return 0


def drops_bomb_near_agent(closest_agent, game_state, action, self):
    """
    Determines if the agent can drop a bomb near the closest agent within the explosion range.
    """
    agent_x, agent_y = game_state['self'][3]
    
    # Check if there's a closest agent
    if closest_agent is None:
        return 0

    target_x, target_y = closest_agent

    # Check if the action is to drop a bomb
    if action == 'BOMB' and game_state['self'][2]:  # Check if bomb can be dropped
        # Check if within range in the y direction
        if abs(target_y - agent_y) <= 3 and target_x == agent_x:
            if agent_x % 2 != 0:  # No wall in between
                return 1

        # Check if within range in the x direction
        if abs(target_x - agent_x) <= 3 and target_y == agent_y:
            if agent_y % 2 != 0:  # No wall in between
                return 1

    return 0


def find_nearest_agent(game_state, action, self):
    """
    Finds the coordinates of the closest agent to the current agent.
    """
    agent_x, agent_y = game_state['self'][3][:2]

    # Generate a list of agent locations: [(x, y), (x, y), ...]
    agent_locations = [item[3][:2] for item in game_state["others"]]

    closest_agent = None
    closest_dist = float('inf')

    # Find the closest agent
    for other_x, other_y in agent_locations:
        dist = np.linalg.norm([other_x - agent_x, other_y - agent_y])
        if dist < closest_dist:
            closest_dist = dist
            closest_agent = (other_x, other_y)

    return closest_agent

def is_bomb_dangerous(bomb, agent_x, agent_y):
    """
    Determines if the bomb poses a danger to the agent based on their coordinates.
    """
    if bomb[0] is None:
        return False

    bomb_x, bomb_y = bomb[0]

    # Check if the agent is within the explosion range vertically or horizontally
    in_y_range = abs(bomb_y - agent_y) <= 3 and bomb_x == agent_x
    in_x_range = abs(bomb_x - agent_x) <= 3 and bomb_y == agent_y

    # Check if the agent is directly on the bomb
    if bomb_x == agent_x and bomb_y == agent_y:
        return True

    # Evaluate conditions for danger
    if in_y_range and (agent_x % 2) != 0:
        return True
    if in_x_range and (agent_y % 2) != 0:
        return True

    return False

def find_dangerous_bombs(game_state, action, self):
    """
    Finds bombs within a dangerous range for the agent and counts how many are dangerous.
    """
    agent_x = game_state['self'][3][0]
    agent_y = game_state['self'][3][1]

    # Generate a list of bombs that are dangerous to the agent
    bomb_infos = game_state["bombs"]
    dangerous_bombs = [bomb for bomb in bomb_infos if is_bomb_dangerous(bomb, agent_x, agent_y)]

    # Sort bombs by proximity to the agent
    dangerous_bombs_sorted = sorted(dangerous_bombs, 
                                     key=lambda bomb: np.linalg.norm([bomb[0][0] - agent_x, bomb[0][1] - agent_y]))

    # Get the closest bombs (up to three)
    closest_bombs = dangerous_bombs_sorted[:3]

    # Count how many of the closest bombs are actually dangerous
    number_of_dangerous_bombs = sum(1 for bomb in closest_bombs if is_bomb_dangerous(bomb, agent_x, agent_y))

    # Keep only the two closest bombs
    dangerous_bombs = closest_bombs[:2]

    return [dangerous_bombs, number_of_dangerous_bombs]

#%%

def drops_bomb_unnecessarily(game_state, action, self):
    """
    Determines if a bomb can be dropped without a specific reason.
    """
    if action == "BOMB":
        closest_crate = find_nearest_crate(game_state, action, self)
        closest_agent = find_nearest_agent(game_state, action, self)

        can_drop_bomb = (
            drops_bomb_near_crate(closest_crate, game_state, action, self) == 0 and
            drops_bomb_near_agent(closest_agent, game_state, action, self) == 0
        )

        if can_drop_bomb:
            return 1

    return 0

def approaches_dangerous_bomb(dangerous_bombs_info, game_state, action, self):
    """
    Checks if the agent is moving towards a dangerous bomb.
    """
    dangerous_bombs = [item[0] for item in dangerous_bombs_info[0]]
    number_of_dangerous_bombs = dangerous_bombs_info[1]

    agent_x = game_state['self'][3][0]
    agent_y = game_state['self'][3][1]

    # Case 1: One dangerous bomb
    if number_of_dangerous_bombs == 1:
        bomb_coord = next((bomb for bomb in dangerous_bombs if bomb is not None), None)

        if bomb_coord is not None:
            bomb_x, bomb_y = bomb_coord

            # Check if moving towards the bomb
            if action == 'UP' and abs(bomb_y - (agent_y - 1)) <= abs(bomb_y - agent_y) and game_state['field'][agent_x][agent_y - 1] == 0:
                return 1
            if action == 'RIGHT' and abs(bomb_x - (agent_x + 1)) <= abs(bomb_x - agent_x) and game_state['field'][agent_x + 1][agent_y] == 0:
                return 1
            if action == 'DOWN' and abs(bomb_y - (agent_y + 1)) <= abs(bomb_y - agent_y) and game_state['field'][agent_x][agent_y + 1] == 0:
                return 1
            if action == 'LEFT' and abs(bomb_x - (agent_x - 1)) <= abs(bomb_x - agent_x) and game_state['field'][agent_x - 1][agent_y] == 0:
                return 1
            if action == 'WAIT':
                return 1

    return 0

def exits_bomb_range(dangerous_bombs_and_number_of_dangerous_bombs, game_state, action, self):
    
    #only coordinates [(x,y), (x,y)]
    dangerous_bombs = [item[0] for item in dangerous_bombs_and_number_of_dangerous_bombs[0]]
    #all info [((x,y),t), ((x,y),t)]
    dangerous_bombs_info = [item for item in dangerous_bombs_and_number_of_dangerous_bombs[0]] 

    number_of_dangerous_bombs = dangerous_bombs_and_number_of_dangerous_bombs[1]

    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]




    #case 1:  one bomb-------------------------------------------------------
    if number_of_dangerous_bombs == 1:

        #get the bomb which is not 'None'
        #get only coordinates (x,y) (no brackets)
        bomb_coord = dangerous_bombs[ [i for i in range(len(dangerous_bombs)) if dangerous_bombs[i] != None][0] ]
        #get all info [((x,y),t)]   (w/ brackets)
        bomb_info = [dangerous_bombs_info[i] for i in [i for i in range(len(dangerous_bombs_info)) if dangerous_bombs_info[i][0] != None]]

        #get the only existing bomb ((x,y),t)  (no brackets)
        bomb_arg = bomb_info[0]
        
        bomb_coord_x = bomb_coord[0]
        bomb_coord_y = bomb_coord[1]

        x, y = bomb_coord_x, bomb_coord_y 



        # the direction which would move agent out of explosion range (if there is no wall or crate)
        # (moving out of range but into a crate trap has no special consideration in this func BUT through 'goes_towards_crate_trap' 
        # func. agent will still be more likely to move out of range AND not into crate trap)
        if action == 'UP':
            if not ( is_bomb_dangerous(bomb_arg, agent_coord_x, agent_coord_y - 1) ) and is_bomb_dangerous(bomb_arg, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y-1] == 0 ):
                return 1

        if action == 'RIGHT':
            if not ( is_bomb_dangerous(bomb_arg, agent_coord_x + 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x+1][agent_coord_y] == 0 ):
                return 1

        if action == 'DOWN':
            if not ( is_bomb_dangerous(bomb_arg, agent_coord_x, agent_coord_y + 1) ) and is_bomb_dangerous(bomb_arg, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x][agent_coord_y+1] == 0 ):
                return 1

        if action == 'LEFT':
            if not ( is_bomb_dangerous(bomb_arg, agent_coord_x - 1, agent_coord_y) ) and is_bomb_dangerous(bomb_arg, agent_coord_x, agent_coord_y) and ( game_state['field'][agent_coord_x-1][agent_coord_y] == 0 ):
                return 1

    return 0  
 
