# Bomberman Reinforcement Learning Agent

## Project Overview

This repository contains the code and report for our Bomberman Reinforcement Learning agent developed as part of the Machine Learning Essentials at the University of Heidelberg. 

In this project, we utilized Q-learning with a focus on feature engineering to train an agent capable of playing Bomberman. Our goal was to create an agent that could successfully navigate the game environment, collect coins, avoid bombs, and strategically place bombs to destroy crates and opponents.

## Team Members

- Dorina Ismaili
- Felix Exner
- Suryansh Chaturvedi

## Repository Contents

- **`callbacks.py`:** Contains the agent's logic, including action selection (`act` function), feature extraction (`extract_features` function), and all the feature functions.
- **`train.py`:**  Implements the training process, handling game events (`game_events_occurred` function), end-of-round updates (`end_of_round` function), and the reward calculation (`calculate_reward` function).
- **`saved_model.pt`:**  A saved model file containing the trained weights of our Q-learning agent. (Add this file after you've trained your model)
- **`report.pdf`:** The project report documenting our approach, experiments, results, and analysis. 

## Results and Analysis
Refer to the report.pdf for detailed information on our approach, experiments, results, and analysis.
