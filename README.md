# Reinforcement Learning (Q-learning, Double Q-learning, Dyna-Q with probabilistic and deterministic models)

## Reference

- Sutton and Barto's book "[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)"
- [Project 7](http://quantsoftware.gatech.edu/Spring_2020_Project_7:_Qlearning_Robot) in the Georgia Tech Spring 2020 course [Machine Learning for Trading](http://quantsoftware.gatech.edu/CS7646_Spring_2020) by Prof. Tucker Balch.

## Characteristics

- The code has been written and tested in Python 3.7.7.
- Q-learning implementation for reinforcement learning problems.
- Options: basic Q-learning, Dyna-Q (for model planning), double Q-learning (to avoid maximization bias).
- Dyna-Q has been implemented with both a deterministic model and a probabilistic model.
- The deterministic model and probabilistic model have both two versions, one using dictionaries (less memory but slower) and one using arrays (more memory but faster).
- Double Q-learning can be used with basic Q-learning as well as with Dyna-Q.
- The Q-learning class in *QLearner.py* can be used for any reinforcement learning problem, while *robot.py* and *test.py* are specific for a grid-world type problem (i.e. finding the best policy to go from a start point to a goal point).
- Note: states must be unique integers in the interval `(0,num_states)`, actions must be unique integers in the interval `(0,num_actions)`, and all states must have all the actions.
- Usage: *python test.py csv-filename*.

## Parameters

`sys.argv[1]` File name with the map layout passed as argument. It must be in a csv file, with the map elements specified using integer numbers.

`map_elements` List of elements allowed in the map layout.

`reward_list` List of rewards associated to each element in `map_elements`.

`move_list` List of allowed moves for the robot (see also an example of an 8-way robot in *test.py*).

`episodes`  Number of episodes (each episode is a trip from start to goal)

`max_steps` Maximum number of steps allowed to reach the goal (for each episode).

`random_rate` Probability the robot will move randomly instead to move as required.

`alpha` Learning rate (used to vary the weight given to new experiences compared with past Q-values).

`gamma` Discount factor (used to progressively reduce the value of future rewards).

`rar` Probability of selecting a random action instead of using the action derived from the Q-table(s) (i.e. probability to explore).

`radr` Rate decay for the probability to explore  (used to reduce the probability to explore with time).

`dyna` Number of simulated updates in Dyna-Q (when equal to zero Dyna-Q is not used).

`model_type` Type of model used for the simulation in Dyna-Q (1-2 are deterministic models, 3-4 are probabilistic models).

`double_Q` Specifies if double Q-learning is used (to avoid maximization bias).

## Examples

All examples are for the map layout in `map.csv`. All initial data are as in *test.py* except when differently specified.

- Basic Q-learning, episodes = 1000, dyna = 0

```
REWARDS:   mean =  -63.1, median =  -32.0, std = 109.8
STEPS:     mean =   62.1, median =   34.0, std =  96.3
Number of updates done:  62085

# # # # # # # # # # # # # # #
#                           #
# S             ~ ~         #
# .           # # # #       #
# . . . .           #     G #
#       . .         #     . #
#         . # # # # # # . . #
#         .         #   .   #
#         . . . . . # . .   #
#       # #       . . .     #
#     # # #                 #
#               # #         #
# # # # # # # # # # # # # # #

BEST PATH: rewards = -22.0, Steps =  24.0
```

- Double Q-learning, episodes = 1000, dyna = 0

```
REWARDS:   mean =  -85.0, median =  -40.0, std = 132.7
STEPS:     mean =   85.5, median =   42.0, std = 130.5
Number of updates done:  85473

# # # # # # # # # # # # # # #
#                           #
# S             ~ ~         #
# .           # # # #       #
# .                 #     G #
# .                 #     . #
# .         # # # # # #   . #
# .                 # . . . #
# . . . . . . .     # .     #
#       # #   . . . . .     #
#     # # #                 #
#               # #         #
# # # # # # # # # # # # # # #

BEST PATH: rewards = -22.0, Steps =  24.0
```

- Double Q-learning, episodes = 50, dyna = 200, model_type = 1

```
REWARDS:   mean =  -70.7, median =  -28.0, std = 158.5
STEPS:     mean =   52.9, median =   30.0, std =  93.5
Number of updates done:  531243

# # # # # # # # # # # # # # #
#                           #
# S . . . .     ~ ~         #
#         .   # # # #       #
#         .         #     G #
#         .         #     . #
#         . # # # # # #   . #
#         .         #   . . #
#         . . .     # . .   #
#       # #   . . . . .     #
#     # # #                 #
#               # #         #
# # # # # # # # # # # # # # #

BEST PATH: rewards = -22.0, Steps =  24.0
```

- Basic Q-learning, episodes = 50, dyna = 200, model_type = 4

```
REWARDS:   mean =  -92.7, median =  -42.5, std = 183.9
STEPS:     mean =   76.9, median =   44.5, std =  94.5
Number of updates done:  567340
Number of updates skipped:  205103

# # # # # # # # # # # # # # #
#                           #
# S             ~ ~         #
# .           # # # #       #
# .                 #   . G #
# . .               #   .   #
#   .       # # # # # # .   #
#   . . . . . . .   # . .   #
#               .   # .     #
#       # #     . . . .     #
#     # # #                 #
#               # #         #
# # # # # # # # # # # # # # #

BEST PATH: rewards = -22.0, Steps =  24.0
```

- Basic Q-learning, episodes = 1000, dyna = 0, but using an 8-way robot

```
REWARDS:   mean =  -66.6, median =  -25.0, std = 120.9
STEPS:     mean =   63.3, median =   27.0, std = 100.1
Number of updates done:  63261

# # # # # # # # # # # # # # #
#                           #
# S             ~ ~         #
#   .         # # # #       #
#     .             #     G #
#       .           #     . #
#         . # # # # # # . . #
#           .       # .     #
#             . .   # .     #
#       # #       . .       #
#     # # #                 #
#               # #         #
# # # # # # # # # # # # # # #

BEST PATH: rewards = -13.0, Steps =  15.0
```
