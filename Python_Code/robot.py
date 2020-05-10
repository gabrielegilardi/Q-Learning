"""
Robot class for a grid-world navigation problem.

Copyright (c) 2020 Gabriele Gilardi
"""

import sys
import numpy as np

class robot:

    def __init__(self, map_layout, map_elements, reward_list, move_list,
                 max_steps=10000, random_rate=0.2):
        """
        map_layout          Map layout (represented using integers)
        map_elements        Elements allowed in the map
        reward_list         Rewards (must correspond to elements in the map)
        move_list           Directions of motion
        random_rate         Probability the robot will move randomly
        max_steps           Max. number of steps for each episode
        """
        self.map_layout = map_layout
        self.map_elements = map_elements
        self.reward_list = reward_list
        self.move_list = move_list
        self.random_rate = random_rate
        self.max_steps = max_steps

        # Initialize internal quantities
        self.n_rows, self.n_cols = self.map_layout.shape    # Map dimensions
        self.num_actions = self.move_list.shape[0]          # Number of actions

    def show_map(self, layout):
        """
        Show the map layout

        0 = " " = empty space
        1 = "#" = wall/obstacle
        2 = "S" = start
        3 = "G" = goal
        4 = "~" = sand
        5 = '.' = trail (only used by the system)
        """
        print()
        print("# " * (self.n_cols+2))
        for row in range(self.n_rows):
            print("#", end=' ')
            for col in range(self.n_cols):
                idx = layout[row, col]
                if (idx == 5):
                    print('.', end=' ')
                else:
                    print(self.map_elements[idx], end=' ')
            print("#")
        print("# " * (self.n_cols+2))
        print()
        return

    def start_pos(self):
        """
        Return the start position "S" as [row,col]
        """
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                if (self.map_layout[row, col] == 2):
                    return np.array([row, col], dtype=int)
        print("Warning: start position not defined (S = 2)")
        sys.exit(1)

    def goal_pos(self):
        """
        Return the goal position "G" as [row,col]
        """
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                if (self.map_layout[row, col] == 3):
                    return np.array([row, col], dtype=int)
        print("Warning: goal position not defined (G = 3)")
        sys.exit(1)

    def move(self, curr_pos, action, curr_map_layout):
        """
        Move the robot by one step and return the new position, the reward,
        and the updated map

        curr_pos            Current robot position
        action              Action to take
        curr_map_layout     Current map layout (before and after this step)
        """
        # Ignore the requested action and choose a random one instead
        # (0 = no random moves, 1 = all moves are random)
        if (np.random.uniform(0.0, 1.0) < self.random_rate):
            action = np.random.randint(0, self.num_actions)

        # New position
        new_pos = curr_pos + self.move_list[action, :]

        # Robot went off the map -> revert back
        if (new_pos[0] < 0 or new_pos[0] >= self.n_rows or new_pos[1] < 0
            or new_pos[1] >= self.n_cols):
            new_pos = curr_pos
            reward = self.reward_list[1]

        # Robot is inside the map -> check the value
        else:
            value_new_pos = self.map_layout[new_pos[0], new_pos[1]]
            reward = self.reward_list[value_new_pos]

            # Moved in an empty space -> keep new position and mark the map
            if (value_new_pos == 0):
                curr_map_layout[curr_pos[0], curr_pos[1]] = 5

            # Hit an obstacle -> revert back
            elif (value_new_pos == 1):
                new_pos = curr_pos

            # In all other cases (2 = back at start, 3 = reached the goal,
            # 4 = moved into sand) -> keep new position but don't mark the map
            elif (value_new_pos >= 2 and value_new_pos <= 4):
                pass

            # Not-allowed element in the map
            else:
                print("Warning: map element not existent")
                sys.exit(1)

        return new_pos, reward, curr_map_layout

    def state(self, curr_pos):
        """
        Convert the robot current position to a unique value (state) in the
        range from 0 to (n_rows x n_cols - 1)
        """
        state = curr_pos[0] * self.n_cols + curr_pos[1]

        return state

    def optimize_path(self, learner, episodes=500):
        """
        Find the optimal path from start to goal position using Q-learning
        and return total rewards and number of steps for each episode
        """
        # Get start (current) and goal position
        start_pos = self.start_pos()
        goal_pos = self.goal_pos()

        # Each episode involves one trip from start to goal
        scores = np.zeros(episodes)
        steps = np.zeros(episodes)
        for episode in range(episodes):

            total_reward = 0.0
            step = 0
            curr_map_layout = self.map_layout.copy()

            # Convert the start position to a state and get the initial action
            curr_pos = start_pos
            curr_state = self.state(curr_pos)
            curr_action = learner.action(curr_state)

            # Iterate until reach the goal or the max number of iterations
            while ((not np.array_equal(curr_pos, goal_pos)) and
                   (step < self.max_steps)):

                # Move to the new position get the corresponding reward
                new_pos, reward, new_map_layout = self.move(curr_pos,
                                                            curr_action,
                                                            curr_map_layout)
                total_reward += reward
                step += 1

                # Convert the new position to a state and update the Q-table
                new_state = self.state(new_pos)
                learner.Q_learning(curr_state, curr_action, new_state, reward)

                # Update position and get the next action
                curr_pos = new_pos
                curr_state = new_state
                curr_action = learner.action(curr_state)
                curr_map_layout = new_map_layout

            # Exited loop due to time-out
            if (step == self.max_steps):
                print("Warning: time-out at episode ", episode)

            # Save scores and steps
            scores[episode] = total_reward
            steps[episode] = step

        return scores, steps

    def best_path(self, learner, start_pos=None):
        """
        Return the optimal path from any start position to the goal position
        If no start position is specified then uses the original start point
        """
        best_map_layout = self.map_layout.copy()

        # Get start position and its state
        original_start_pos = self.start_pos()
        if (start_pos is None):
            # Use original start position if none specified
            start_pos = original_start_pos
        else:
            # Use specified start position
            start_pos = np.asarray(start_pos)
            # Mark the new start position and un-mark the original one
            best_map_layout[start_pos[0], start_pos[1]] = 2
            best_map_layout[original_start_pos[0], original_start_pos[1]] = 0
        curr_state = self.state(start_pos)

        # Get end position and its state
        end_pos = self.goal_pos()
        end_state = self.state(end_pos)

        # Loop from start to goal or up to max. count
        step = 0
        curr_pos = start_pos
        total_reward = 0
        while ((curr_state != end_state) and (step < self.max_steps)):

            # Get best action for current state
            action = learner.best_action(curr_state)

            # Move robot
            new_pos = curr_pos + self.move_list[action, :]

            # Robot went off the map
            if (new_pos[0] < 0 or new_pos[0] >= self.n_rows or new_pos[1] < 0
                or new_pos[1] >= self.n_cols):
                print("Warning: tried to move in ", new_pos)
                print("Last position = ", curr_pos, " State = ", curr_state)
                return best_map_layout, total_reward, step

            # Robot is inside the map -> check the value
            else:
                value_new_pos = self.map_layout[new_pos[0], new_pos[1]]
                reward = self.reward_list[value_new_pos]

                # Moved in an empty space -> mark the map
                if (value_new_pos == 0):
                    best_map_layout[new_pos[0], new_pos[1]] = 5

                # Hit an obstacle
                elif (value_new_pos == 1):
                    print("Warning: tried to move in ", new_pos)
                    print("Last position = ", curr_pos, " State = ", curr_state)
                    return best_map_layout, total_reward, step

                # In all other cases (2 = back in start position, 3 = reached
                # the goal, 4 = moved into sand) -> don't mark the map
                elif (value_new_pos >= 2 and value_new_pos <= 4):
                    pass

                # Not-allowed element in the map
                else:
                    print("Warning: map element not existent")
                    sys.exit(1)

            # Update for next step
            curr_pos = new_pos
            curr_state = self.state(curr_pos)
            step += 1
            total_reward += reward

        # Exited loop due to the time-out
        if (step == self.max_steps):
            print("Warning: could not find the goal before time-out")

        return best_map_layout, total_reward, step
