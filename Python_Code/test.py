
"""
================================================================================
Reinforcement Learning (Q-learning, Double Q-learning, Dyna-Q with probabilistic
and deterministic models)
================================================================================

References
==========
- Based on project 7 in the Georgia Tech Spring 2020 course "Machine Learning
  for Trading" by Prof. Tucker Balch.
- Course: http://quantsoftware.gatech.edu/CS7646_Spring_2020
- Project: http://quantsoftware.gatech.edu/Spring_2020_Project_7:_Qlearning_Robot
- Main book reference: Sutton and Barto, "Reinforcement Learning: An Introduction"
  (http://incompleteideas.net/book/the-book-2nd.html)

Characteristics
===============
- The code has been written and tested in Python 3.7.7.
- Q-learning implementation for reinforcement learning problems.
- Options: basic Q-learning, Dyna-Q (for model planning), double Q-learning (to
  avoid maximization bias).
- Dyna-Q has been implemented with both a deterministic model and a probabilistic
  model.
- The deterministic model and probabilistic model have both two versions, one
  using dictionaries (less memory but slower) and one using arrays (more memory
  but faster).
- Double Q-learning can be used with basic Q-learning as well as with Dyna-Q.
- The Q-learning class in <QLearner.py> can be used for any reinforcement learning
  problem, while <robot.py> and <test.py> are specific for a grid-world type of
  problem (i.e. finding the best policy to go from a start point to a goal point).
- Usage: python test.py <csv-filename>.

Parameters
===============
sys.argv[1]
    File name with the map layout passed as argument. It must be in a csv file,
    with the map elements specified using integer numbers.
map_elements
    List of elements allowed in the map layout.
reward_list
    List of rewards associated to each element in <map_elements>.
move_list
    List of allowed moves for the robot.
episodes
    Number of episodes (each episode is a trip from start to goal)
max_steps
    Maximum number of steps allowed to reach the goal (for each episode).
0 <= random_rate <= 1
    Probability the robot will move randomly instead to move as required.
0 <= alpha <= 1
    Learning rate (used to vary the weight given to new experiences compared with
    past Q-values).
0 <= gamma <= 1
    Discount factor (used to progressively reduce the value of future rewards).
0 <= rar <= 1
    Probability of selecting a random action instead of using the action derived
    from the Q-table(s) (i.e. probability to explore).
0 <= radr <= 1
    Rate decay for the probability to explore  (used to reduce the probability to
    explore with time).
dyna >= 0
    Number of simulated updates in Dyna-Q (when equal to zero Dyna-Q is not used).
model_type = 1, 2, 3, 4
    Type of model used for the simulation in Dyna-Q (1-2 are deterministic models,
    3-4 are probabilistic models).
double_Q = True, False
    Specifies if double Q-learning is used (to avoid maximization bias).

Examples
========
All examples are for the map layout in `map.csv`. All initial data are as in this
file, except when differently specified.

- Basic Q-learning, episodes = 1000, dyna = 0

        REWARDS:   mean =  -63.1, median =  -32.0, std = 109.8
        STEPS:     mean =   62.1, median =   34.0, std =  96.3
        Number of updates done:  62085
        BEST PATH: rewards = -22.0, Steps =  24.0

- Double Q learning, episodes = 1000, dyna = 0

        REWARDS:   mean =  -85.0, median =  -40.0, std = 132.7
        STEPS:     mean =   85.5, median =   42.0, std = 130.5
        Number of updates done:  85473
        BEST PATH: rewards = -22.0, Steps =  24.0

- Double Q-learning, episodes = 50, dyna = 200, model_type = 1

        REWARDS:   mean =  -70.7, median =  -28.0, std = 158.5
        STEPS:     mean =   52.9, median =   30.0, std =  93.5
        Number of updates done:  531243
        BEST PATH: rewards = -22.0, Steps =  24.0

- Basic Q-learning, episodes = 50, dyna = 200, model_type = 4

        REWARDS:   mean =  -92.7, median =  -42.5, std = 183.9
        STEPS:     mean =   76.9, median =   44.5, std =  94.5
        Number of updates done:  567340
        Number of updates skipped:  205103
        BEST PATH: rewards = -22.0, Steps =  24.0

- Basic Q-learning, episodes = 1000, dyna = 0, but using an 8-way robot

        REWARDS:   mean =  -66.6, median =  -25.0, std = 120.9
        STEPS:     mean =   63.3, median =   27.0, std = 100.1
        Number of updates done:  63261
        BEST PATH: rewards = -13.0, Steps =  15.0
"""

import sys
import numpy as np

import QLearner as ql
import robot as rb

# Elements allowed in the map
map_elements = [' ',        # 0 = empty space
                '#',        # 1 = wall/obstacle
                'S',        # 2 = start (must be defined)
                'G',        # 3 = goal (must be defined)
                '~']        # 4 = sand

# Rewards (must correspond to elements in the map)
reward_list = np.array([-1.0,       # empty space
                        -1.0,       # wall/obstacle
                        -1.0,       # start (walk-back)
                        +1.0,       # goal
                        -100.0])    # sand

# Directions of motion (4-way robot)
move_list = np.array([[-1,  0],         # Go North one step
                      [ 0, +1],         # Go East one step
                      [+1,  0],         # Go South one step
                      [ 0, -1]])        # Go West one step

# Directions of motion (8-way robot)
# move_list = np.array([[-1,  0],         # Go North one step
#                       [-1, +1],         # Go North-East one step
#                       [ 0, +1],         # Go East one step
#                       [+1, +1],         # Go South-East one step
#                       [+1,  0],         # Go South one step
#                       [+1, -1],         # Go South-West one step
#                       [ 0, -1],         # Go West one step
#                       [-1, -1]])        # Go North-West one step

# Other grid-world parameters
episodes = 1000         # Number of episodes
max_steps = 10000 	    # Max. number of steps for each episode
random_rate = 0.2       # Probability the robot will move randomly

# Q-learner parameters
alpha = 0.2             # Learning rate
gamma = 0.9             # Discount factor
rar = 0.50              # Probability to explore
radr = 0.99             # Rate decay for the probability to explore
dyna = 0	            # Number of simulated updates in Dyna-Q (not used if zero)
model_type = 1          # Type of model used for the simulation in Dyna-Q
                        # 1 = Deterministic model (T and R defined as dictionaries)
                        # 2 = Deterministic model (T and R defined as arrays)
                        # 3 = Probabilistic model (T and R defined as dictionaries)
                        # 4 = Probabilistic model (T and R defined as arrays)
double_Q = False        # True = use double Q-learning
                        # False = don't use double Q-learning


# ======= Main Code ======= #

np.random.seed(1)

# Read the map layout from the csv file specified on the command line
if (len(sys.argv) != 2):
    print("Usage: python test.py <csv-filename>")
    sys.exit(1)
map_layout = np.asarray(np.loadtxt(sys.argv[1], delimiter=','), dtype=int)

# Initialize robot and map quantities
bot = rb.robot(map_layout, map_elements, reward_list, move_list, max_steps=max_steps,
               random_rate=random_rate)

# Initialize the Q-learner
num_states = map_layout.size
num_actions = move_list.shape[0]
learner = ql.QLearner(num_states, num_actions, alpha=alpha, gamma=gamma, rar=rar,
                      radr=radr, dyna=dyna, double_Q=double_Q, model_type=model_type)

# Build the Q-table(s)
scores, steps = bot.optimize_path(learner, episodes)

# Print results
print()
print("REWARDS:   mean = {0:6.1f}, median = {1:6.1f}, std = {2:5.1f}"
      .format(np.mean(scores), np.median(scores), np.std(scores)))
print("STEPS:     mean = {0:6.1f}, median = {1:6.1f}, std = {2:5.1f}"
      .format(np.mean(steps), np.median(steps), np.std(steps)))
print("Number of updates done: ", learner.count_update_Q)
if (dyna > 0 and (model_type == 2 or model_type == 4)):
    print("Number of updates skipped: ", learner.count_skip)

# Print best map and corresponding rewards and steps
best_map, best_reward, best_step = bot.best_path(learner)
bot.show_map(best_map)
print("BEST PATH: rewards = {0:5.1f}, Steps = {1:5.1f}".
      format(best_reward, best_step))
