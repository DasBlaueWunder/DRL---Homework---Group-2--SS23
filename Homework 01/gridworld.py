"""Implementation of a simple gridworld environment and monte carlo evaluation."""

from collections import defaultdict
import random
import time
import numpy as np


class GridWorld:
    """A simple gridworld environment.

    Attributes:
        size (int, int): The size of the gridworld.
        start (int, int): The starting position of the agent.
        goal (int, int): The goal position of the agent.
        state (int, int): The current position of the agent.
        walls [(int, int)]: The positions of the walls in the gridworld.
        winds [(int, int, int, float)]: The positions. direction and strength of the wind.
            - 0 is up
            - 1 is right
            - 2 is down
            - 3 is left
    """

    def __init__(
        self,
        size: (int, int) = (4, 3),
        start: (int, int) = (0, 0),
        goal: (int, int) = (3, 2),
        walls: [(int, int)] = None,
        winds: [(int, int, int, float)] = None,
    ):
        self.size = size
        self.start = start
        self.goal = goal
        self.state = start
        self.state = start

        # Mutable default arguments are bad practice, that's why we do this
        if walls is None:
            self.walls = [(1, 1), (2, 1)]
        else:
            self.walls = walls
        if winds is None:
            self.winds = [(2, 2, 3, 0.5)]
        else:
            self.winds = winds

        # Check if the start and goal are valid
        if start in self.walls:
            raise ValueError("Start position must not be a wall.")
        if goal in self.walls:
            raise ValueError("Goal position must not be a wall.")
        if start == goal:
            raise ValueError("Start and goal positions must be different.")
        if start[0] < 0 or start[0] >= self.size[0]:
            raise ValueError("Start position must be within the gridworld.")
        if start[1] < 0 or start[1] >= self.size[1]:
            raise ValueError("Start position must be within the gridworld.")
        if goal[0] < 0 or goal[0] >= self.size[0]:
            raise ValueError("Goal position must be within the gridworld.")
        if goal[1] < 0 or goal[1] >= self.size[1]:
            raise ValueError("Goal position must be within the gridworld.")

    def main_loop(self):
        """Run a main loop for the environment (1 episode)."""
        self.reset()
        self.render()
        while True:
            action = self.policy(self.state)
            self.state = self.state_transition(self.state, action)
            self.render(action)
            if self.is_terminal_state(self.state):
                break

    def reset(self):
        """Reset the environment to its initial state."""
        self.state = self.start

    def action_space(self) -> [int]:
        """Return the action space of the environment.

        Actions are represented as integers:
            - 0 is up
            - 1 is right
            - 2 is down
            - 3 is left
        """
        return [0, 1, 2, 3]

    def get_next_state(self, state: (int, int), action: int) -> (int, int):
        """Return the next state of the environment given the current state and action.

        Args:
            state (int, int): The current state of the environment.
            action (int): The action to take in the current state.

        Returns:
            (int, int): The next state of the environment.
        """
        if action == 0:
            return (state[0], state[1] - 1)
        if action == 1:
            return (state[0] + 1, state[1])
        if action == 2:
            return (state[0], state[1] + 1)
        if action == 3:
            return (state[0] - 1, state[1])
        raise ValueError("Action must be 0, 1, 2, or 3.")

    def is_terminal_state(self, state: (int, int)) -> bool:
        """Return whether the given state is terminal.

        Args:
            state (int, int): The state to check.

        Returns:
            bool: Whether the given state is terminal.
        """
        return state == self.goal

    def state_transition(self, state: (int, int), action: int) -> (int, int):
        """Return the next state of the environment given the current state and action.

        Args:
            state (int, int): The current state of the environment.
            action (int): The action to take in the current state.

        Returns:
            (int, int): The next state of the environment.
        """
        next_state = self.get_next_state(state, action)
        # if the next state is a wall, stay in the same state
        if next_state in self.walls:
            return state
        for wind in self.winds:
            # if the next state is in the wind...
            if next_state == (wind[0], wind[1]):
                # blown away by the wind in the given direction, with a probability of wind[3]
                if random.random() < wind[3]:
                    next_state = self.get_next_state(next_state, wind[2])
        # if the next state would be out of bounds, stay in the same state
        if next_state[0] < 0 or next_state[0] >= self.size[0]:
            return state
        if next_state[1] < 0 or next_state[1] >= self.size[1]:
            return state
        return next_state

    def policy(self, state: (int, int)) -> int:
        """Return the action to take in the given state.

        A simple policy, where the agent walks in the direction of the goal with
        a probability of 0.8 and otherwise acts randomly. We could leave out the
        cases where two actions are equally good, but this would bias the agent
        towards going right first.

        Args:
            state (int, int): The state to take an action in.

        Returns:
            int: The action to take in the given state.
        """
        if random.random() < 0.2:
            return random.choice(self.action_space())
        # Cases, where two actions are equally good
        if state[0] < self.goal[0] and state[1] < self.goal[1]:
            return random.choice([1, 2])
        if state[0] < self.goal[0] and state[1] > self.goal[1]:
            return random.choice([1, 0])
        if state[0] > self.goal[0] and state[1] < self.goal[1]:
            return random.choice([3, 2])
        if state[0] > self.goal[0] and state[1] > self.goal[1]:
            return random.choice([3, 0])
        # Cases, where only one action leads to the goal
        if state[0] < self.goal[0]:
            return 1
        if state[0] > self.goal[0]:
            return 3
        if state[1] < self.goal[1]:
            return 2
        if state[1] > self.goal[1]:
            return 0

    def reward(self, state: (int, int)) -> float:
        """Return the reward for the given state.

        Args:
            state (int, int): The state to return the reward for.

        Returns:
            float: The reward for the given state.
        """
        return 1 if state == self.goal else -0.01

    def first_visit_mc_evaluation(self, num_iterations: int = 1000) -> np.ndarray:
        """Perform first-visit Monte Carlo evaluation.

        Args:
            num_iterations (int): The number of iterations to perform.

        Returns:
            np.ndarray: The state-value function.
        """
        # Initialize the value function to zero for all states
        value_function = np.zeros(self.size)
        # value_function = np.full(self.size, 0.1)
        # When a state is encountered for the first time, defaultdict will create a new list for it
        all_returns = defaultdict(list)
        for _ in range(num_iterations):
            self.reset()
            # episode is a list of (state, action, reward) tuples
            episode = []
            # Generate an episode
            while True:
                action = self.policy(self.state)
                next_state = self.state_transition(self.state, action)
                reward = self.reward(next_state)
                episode.append((self.state, action, reward))
                self.state = next_state
                if self.is_terminal_state(self.state):
                    episode.append((self.state, None, 0))
                    break

            # Keep track of visited states
            visited = set()
            # For each tuple in the episode
            for j, (state, action, reward) in enumerate(episode):
                # If the state has not been visited before
                if state not in visited:
                    # Add it to the visited set
                    visited.add(state)
                    # Calculate the concrete return for the state (sum of rewards from the
                    # state to the end of the episode)
                    concrete_return = sum([x[2] for x in episode[j:]])
                    # Add the concrete return to the list of returns for the state
                    all_returns[state].append(concrete_return)
                    # Calculate the mean of all returns for the state and update the value function
                    value_function[state] = np.mean(all_returns[state])

        # Check if the goal state was visited in the episode and update its value to 1.0
        if self.goal in visited:
            value_function[self.goal] = 1.0

        # Return the value function (transposed to match the gridworld)
        return value_function.T

    # --------------------------------------------------------------------------
    # The rest of the code is for visualization purposes only
    # --------------------------------------------------------------------------

    def render(self, action: int = None):
        """Render the environment."""
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                if (x, y) == self.state:
                    current_state = True
                else:
                    current_state = False

                if current_state:
                    if action is not None:
                        if action == 0:
                            print("↑", end="")
                        elif action == 1:
                            print("→", end="")
                        elif action == 2:
                            print("↓", end="")
                        elif action == 3:
                            print("←", end="")
                        else:
                            print("X", end="")
                    else:
                        print("X", end="")
                elif (x, y) == self.goal:
                    print("G", end="")
                elif (x, y) == self.start:
                    print("S", end="")
                elif (x, y) in self.walls:
                    print("W", end="")
                elif (x, y) in [wind[:2] for wind in self.winds]:
                    # check the direction of the wind
                    for wind in self.winds:
                        if (x, y) == wind[:2]:
                            direction = wind[2]
                            if direction == 0:
                                print("^", end="")
                            elif direction == 1:
                                print(">", end="")
                            elif direction == 2:
                                print("v", end="")
                            elif direction == 3:
                                print("<", end="")
                            break
                else:
                    print(".", end="")
            print("")
        print("")
        time.sleep(2)

    def visualize_value_function(self, value_function) -> None:
        """Print the value function in a readable format."""
        values_list = value_function.tolist()
        for row in values_list:
            row = " ".join([f"{val:.4f}" for val in row])
            print(row)
        print()


if "__main__" == __name__:
    env = GridWorld(winds=[(2, 2, 3, 0.9)])
    print("\n" + "Initial state, representation of the environment")
    print("S: start, G: goal, W: wall, ^>v<: wind direction \n")
    env.render()

    # Example run (with visualization)
    # env.main_loop()

    # print("\n" + "MC estimation after 50 episodes, wind strength 0.90")
    # V = env.first_visit_mc_evaluation(50)
    # env.visualize_value_function(V)

    # print("\n" + "MC estimation after 200 episodes")
    # V = env.first_visit_mc_evaluation(200)
    # env.visualize_value_function(V)

    # print("\n" + "MC estimation after 500 episodes")
    # V = env.first_visit_mc_evaluation(500)
    # env.visualize_value_function(V)

    # print("\n" + "MC estimation after 1000 episodes")
    # V = env.first_visit_mc_evaluation(1000)
    # env.visualize_value_function(V)

    # print("\n" + "MC estimation after 10000 episodes")
    # V = env.first_visit_mc_evaluation(10000)
    # env.visualize_value_function(V)
