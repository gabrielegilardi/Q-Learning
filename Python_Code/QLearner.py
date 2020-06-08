"""
QLearner class for reinforcement learning using Q-table(s)

Copyright (c) 2020 Gabriele Gilardi

Notes:
- states must be unique integers in the interval (0,num_states)
- action must be unique integers in the interval (0,num_actions)
- all states must have all the actions
"""

import numpy as np

class QLearner:

    def __init__(self, num_states, num_actions, alpha=0.2, gamma=0.9, rar=0.5,
                 radr=0.99, dyna=0, model_type=1, double_Q=False):
        """
        num_states      Number of states
        num_actions     Number of actions
        alpha           Learning rate (used to vary the weight given to new
                        experiences compared with past Q-values)
        gamma           Discount factor (used to progressively reduce the value
                        of future rewards)
        rar             Probability of selecting a random action (probability
                        to explore)
        radr            Rate decay for the probability to explore
        dyna            Number of simulated updates in Dyna-Q
        double_Q        Specifies if double Q-learning is used (to avoid
                        maximization bias)
        model_type      Type of model used for the simulation in Dyna-Q
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.double_Q = double_Q
        self.model_type = model_type

        # Initialize Q-table(s)
        self.count_update_Q = 0
        if(self.double_Q):      # Double Q-learning
            self.Q1 = np.zeros((num_states, num_actions))
            self.Q2 = np.zeros((num_states, num_actions))
        else:                   # Single Q-learning
            self.Q = np.zeros((num_states, num_actions))

        # Initialize Dyna
        if (dyna > 0):
            # Deterministic model (T and R defined as dictionaries)
            if (model_type == 1):
                self.TR = {}
            # Deterministic model (T and R defined as arrays)
            elif (model_type == 2):
                self.T = np.full((num_states, num_actions), -1, dtype=int)
                self.R = np.zeros((num_states, num_actions))
                self.count_skip = 0
            # Probabilistic model (T and R defined as dictionaries)
            elif (model_type == 3):
                self.T = {}
                self.R = {}
            # 4 = Probabilistic model (T and R defined as arrays)
            elif (model_type == 4):
                self.T = np.zeros((num_states, num_actions, num_states), dtype=int)
                self.R = np.zeros((num_states, num_actions))
                self.count_skip = 0

    def action(self, s):
        """
        Returns an action using epsilon-greedy
        """
        # Exploration
        if (np.random.uniform(0.0, 1.0) < self.rar):
            a = np.random.randint(0, self.num_actions)

        # Exploitation
        else:
            if (self.double_Q):         # Double Q-learning
                a = np.argmax(self.Q1[s, :] + self.Q2[s, :])

            else:                       # Single Q-learning
                a = np.argmax(self.Q[s, :])

        # After each update reduce the chance of exploration
        self.rar = self.rar * self.radr

        return a

    def Q_learning(self, s, a, s_prime, r):
        """
        Manages the different Q-learning options
        """
        # Update Q-table(s) (direct reinforcement learning)
        self.update_Q(s, a, s_prime, r)

        # Apply Dyna-Q (model learning and planning)
        if (self.dyna > 0):

            # Simulate as deterministic model (T and R defined as dictionaries)
            if (self.model_type == 1):
                self.model1_deterministic(s, a, s_prime, r)

            # Simulate as deterministic model (T and R defined as arrays)
            elif (self.model_type == 2):
                self.model2_deterministic(s, a, s_prime, r)

            # Simulate as probabilistic model (T and R defined as dictionaries)
            elif (self.model_type == 3):
                self.model3_probabilistic(s, a, s_prime, r)

            # Simulate as probabilistic model (T and R defined as arrays)
            elif (self.model_type == 4):
                self.model4_probabilistic(s, a, s_prime, r)

    def update_Q(self, s, a, s_prime, r):
        """
        Updates Q-table(s)
        """
        self.count_update_Q += 1

        # Double Q
        if (self.double_Q):

            # Update Q1 using argmax(Q2)
            if (np.random.uniform(0.0, 1.0) < 0.5):
                a1 = np.argmax(self.Q1[s_prime, :])
                delta = r + self.gamma * self.Q2[s_prime, a1] - self.Q1[s, a]
                self.Q1[s, a] += self.alpha * delta

            # Update Q2 using argmax(Q1)
            else:
                a2 = np.argmax(self.Q2[s_prime, :])
                delta = r + self.gamma * self.Q1[s_prime, a2] - self.Q2[s, a]
                self.Q2[s, a] += self.alpha * delta

        # Single Q
        else:
            delta = r + self.gamma * np.amax(self.Q[s_prime, :]) - self.Q[s, a]
            self.Q[s, a] += self.alpha * delta

    def model1_deterministic(self, s, a, s_prime, r):
        """
        Simulates as deterministic model (T and R defined as dictionaries)
        """
        # Add <new state,reward> to the model as the result of <state,action>
        if (s not in self.TR.keys()):
            self.TR[s] = {}
        self.TR[s][a] = (s_prime, r)

        # Simulate the model (planning)
        for i in range(self.dyna):

            # Randomly pick a state <s_sim> from previously observed states
            idx = np.random.randint(0, len(self.TR))
            s_sim = list(self.TR.keys())[idx]

            # Randomly pick an action previously taken in state <s_sim>
            idx = np.random.randint(0, len(self.TR[s_sim]))
            a_sim = list(self.TR[s_sim].keys())[idx]

            # Determine next state and reward using the model as defined by TR
            s_prime_sim, r_sim = self.TR[s_sim][a_sim]

            # Update Q-table(s)
            self.update_Q(s_sim, a_sim, s_prime_sim, r_sim)

    def model2_deterministic(self, s, a, s_prime, r):
        """
        Simulates as deterministic model (T and R defined as arrays)
        """
        # Add <new state,reward> to the model as the result of <state,action>
        self.T[s, a] = s_prime
        self.R[s, a] = r

        # Simulate the model (planning)
        for i in range(self.dyna):

            # Randomly pick a state <s_sim> from ALL possible states
            s_sim = np.random.randint(0, self.num_states)

            # Randomly pick an action <a_sim> from all possible actions
            a_sim = np.random.randint(0, self.num_actions)

            # Determine next state and reward using the model defined by T and R
            s_prime_sim = self.T[s_sim, a_sim]
            r_sim = self.R[s_sim, a_sim]

            # Skip the update if the <state-action> pair has not been previously
            # visited
            if (s_prime_sim == -1):
                self.count_skip += 1
                continue

            # Update Q-table(s)
            self.update_Q(s_sim, a_sim, s_prime_sim, r_sim)

    def model3_probabilistic(self, s, a, s_prime, r):
        """
        Simulates as probabilistic model (T and R defined as dictionaries)
        """
        # Add <new state,reward> to the model as the result of <state,action>
        if (s not in self.T.keys()):
            self.T[s] = {}
            self.R[s] = {}
        if (a not in self.T[s].keys()):
            self.T[s][a] = {}
            self.R[s][a] = 0.0
        if (s_prime not in self.T[s][a].keys()):
            self.T[s][a][s_prime] = 0
        self.T[s][a][s_prime] += 1
        self.R[s][a] += self.alpha * (r - self.R[s][a])

        # Simulate the model (planning)
        for i in range(self.dyna):

            # Randomly pick a state <s_sim> from previously observed states
            idx = np.random.randint(0, len(self.T))
            s_sim = list(self.T.keys())[idx]

            # Randomly pick an action previously taken in state <s_sim>
            idx = np.random.randint(0, len(self.T[s_sim]))
            a_sim = list(self.T[s_sim].keys())[idx]

            # Determine next state and reward using the model defined by T and R
            idx = np.argmax(np.asarray(list(self.T[s_sim][a_sim].values())))
            s_prime_sim = list(self.T[s_sim][a_sim].keys())[idx]
            r_sim = self.R[s_sim][a_sim]

            # Update Q-table(s)
            self.update_Q(s_sim, a_sim, s_prime_sim, r_sim)

    def model4_probabilistic(self, s, a, s_prime, r):
        """
        Simulates as probabilistic model (T and R defined as arrays)
        """
        # Add <new state,reward> to the model as the result of <state,action>
        self.T[s, a, s_prime] += 1
        self.R[s, a] += self.alpha * (r - self.R[s, a])

        # Simulate the model (planning)
        for i in range(self.dyna):

            # Randomly pick a state <s_sim> from ALL possible states
            s_sim = np.random.randint(0, self.num_states)

            # Randomly pick an action <a_sim> from all possible actions
            a_sim = np.random.randint(0, self.num_actions)

            # Determine next state and reward using the model defined by T and R
            s_prime_sim = np.argmax(self.T[s_sim, a_sim, :])
            r_sim = self.R[s_sim, a_sim]

            # Skip the update if the state has not been previously visited (if
            # the pair <s_sim,a_sim> has been visited before then at least one
            # value in T[s_sim,a_sim,:] must be larger than zero)
            if (self.T[s_sim, a_sim, s_prime_sim] == 0):
                self.count_skip += 1
                continue

            # Update Q-table(s)
            self.update_Q(s_sim, a_sim, s_prime_sim, r_sim)

    def best_action(self, s):
        """
        Returns the best action for the specified state (greedy)
        """
        # Double Q-learning
        if (self.double_Q):
            a = np.argmax(self.Q1[s, :] + self.Q2[s, :])

        # Single Q-learning
        else:
            a = np.argmax(self.Q[s, :])

        return a
