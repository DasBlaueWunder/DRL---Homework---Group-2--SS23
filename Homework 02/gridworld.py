"""Implementation of a simple gridworld environment and monte carlo evaluation."""

from collections import defaultdict
import random
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


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
        step_cost (float): The cost of taking a step in the gridworld.
    """

    def __init__(
        self,
        size: (int, int) = (4, 3),
        start: (int, int) = (0, 0),
        goal: (int, int) = (3, 2),
        walls: [(int, int)] = None,
        winds: [(int, int, int, float)] = None,
        step_cost: float = -0.1,
    ):
        self.size = size
        self.start = start
        self.goal = goal
        self.state = start
        self.step_cost = step_cost

        # All possible states in the gridworld (implied by size)
        self.states = [(x, y) for x in range(size[0]) for y in range(size[1])]

        # Mutable default arguments are bad practice, that's why we do this
        self.walls = walls or [(1, 1), (2, 1)]
        self.winds = winds or [(2, 2, 3, 0.5)]

        # Check if the start and goal are valid
        if start in self.walls:
            raise ValueError("Start position must not be a wall.")
        if goal in self.walls:
            raise ValueError("Goal position must not be a wall.")
        if start == goal:
            raise ValueError("Start and goal positions must be different.")
        if not self._is_within_bounds(start):
            raise ValueError("Start position must be within the gridworld.")
        if not self._is_within_bounds(goal):
            raise ValueError("Goal position must be within the gridworld.")

        print("\nThis is our Gridworld:\n")
        self.render()

    def _is_within_bounds(self, state: (int, int)) -> bool:
        """Check if a state is within the bounds of the gridworld."""
        x, y = state
        return 0 <= x < self.size[0] and 0 <= y < self.size[1]

    def main_loop(self):
        """Run a main loop for the environment (1 episode)."""
        self.reset()
        self.render()
        while True:
            action = self.example_policy(self.state)
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

    def example_policy(self, state: (int, int)) -> int:
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
        return 1 if state == self.goal else self.step_cost

    def first_visit_mc_control(
        self,
        num_iterations: int = 100,
        epsilon: float = 0.1,
        gamma: float = 0.9,
        plot: str = None,
    ) -> None:
        """Perform on-policy first-visit Monte Carlo control, using epsilon-greedy exploration.

        We start by creating and arbitrary epsilon-soft policy. Then, we initialize the
        state-action value function to arbitrary values for each state-action pair. We also
        create a dictionary that maps each state-action pair to a list of returns. Then, for
        each iteration, we generate an episode using the epsilon-soft policy, and update the
        state-action value function using the first-visit Monte Carlo method. Finally, we
        update the policy to be greedy with respect to the state-action value function.

            Args:
                num_iterations (int): The number of iterations to perform.
                epsilon (float): The probability of taking a random action.
                gamma (float): The discount factor.
                plot (str):
                    None: Do not plot anything.
                    "time": Plot the mean return over the amount of seconds it took.
                    "iteration": Plot the mean return over the amount of iterations.
        """
        # Initialize an arbitrary epsilon-soft policy. It is a dictionary that maps each state to a probability distribution over actions
        policy = {}
        for state in self.states:
            for action in self.action_space():
                policy[(state, action)] = 1 / len(self.action_space())
        # Initialize the state-action value function to arbitrary values for each state-action pair
        q = {}
        for state in self.states:
            for action in self.action_space():
                q[(state, action)] = 0
        # Empty list of returns for each state-action pair
        returns = defaultdict(list)

        # Initialize an empty list to store the average return per episode
        average_returns = []

        # Initialize an empty list to store the time per episode
        episode_times = []

        # Store start time
        start_time = time.time()

        # For each iteration...
        counter = 0
        for _ in tqdm(range(num_iterations)):
            if counter == 10:
                self.print_policy(policy)
                time.sleep(0.5)
                counter = 0

            try:  # Catch KeyboardInterrupt
                # Generate an episode using the epsilon-soft policy
                self.reset()
                episode = []
                while True:
                    # Choose an action from the action space, based on its probability in the policy
                    action = np.random.choice(
                        self.action_space(),
                        p=[policy[(self.state, a)] for a in self.action_space()],
                    )
                    next_state = self.state_transition(self.state, action)
                    reward = self.reward(next_state)
                    episode.append((self.state, action, reward))
                    self.state = next_state
                    if self.state == self.goal:
                        break

                # Update the state-action value function using the first-visit Monte Carlo method
                visited = set()
                G = 0
                # For each tuple in the episode
                for j, (state, action, reward) in enumerate(episode):
                    # If the state has not been visited before
                    if (state, action) not in visited:
                        # Add it to the visited set
                        visited.add((state, action))
                        # Calculate the concrete return for the state (sum of rewards from the
                        # state to the end of the episode)
                        G = sum([x[2] * gamma**i for i, x in enumerate(episode[j:])])
                        # Add the concrete return to the list of returns for the state
                        returns[(state, action)].append(G)
                        # Calculate the mean of all returns for the state and update the value function
                        q[(state, action)] = np.mean(returns[(state, action)])

                # Update the policy to be greedy with respect to the state-action value function
                for state, _, _ in episode:
                    # Select action greedily with respect to the state-action value function
                    best_action = np.argmax(
                        [q[(state, action)] for action in self.action_space()]
                    )

                    # Update the policy (epsilon-soft)
                    for action in self.action_space():
                        if action == self.action_space()[best_action]:
                            policy[(state, action)] = (
                                1 - epsilon + epsilon / len(self.action_space())
                            )
                        else:
                            policy[(state, action)] = epsilon / len(self.action_space())

                # Store the average return of this episode
                average_return = sum([x[2] for x in episode]) / len(episode)
                average_returns.append(average_return)

                # Store the time of this episode
                episode_times.append(time.time() - start_time)

                counter += 1
            except KeyboardInterrupt:
                break
        # Store end time
        end_time = time.time()

        # Calculate the number of seconds it took to train the agent
        training_time = end_time - start_time

        print(f"Training finished after {num_iterations} iterations. Resulting policy:")
        self.print_policy(policy)
        if plot:
            if plot == "iteration":
                # Plot the returns obtained in each episode
                plt.plot(range(num_iterations), average_returns)
                plt.xlabel("Iteration")
            elif plot == "time":
                # Plot the returns obtained in each episode
                plt.plot(episode_times, average_returns)
                plt.xlabel("Time (seconds)")
            plt.ylabel("Mean return-per-episode")
            plt.show()

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
                action = self.example_policy(self.state)
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

    def print_policy(self, policy: dict) -> None:
        """Visualize the policy, i.e. the action to take in each state.

        For each state, the action with the highest value is selected. The state
        space is represented by the gridworld and the actions are represented by
        arrows.

            Args:
                policy (dict): The policy.
        """
        print("\n")
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                if (x, y) == self.goal:
                    print("G", end="")
                elif (x, y) in self.walls:
                    print("W", end="")
                else:
                    best_action = np.argmax(
                        [policy[(x, y), a] for a in range(len(self.action_space()))]
                    )
                    if best_action == 0:
                        print("↑", end="")
                    elif best_action == 1:
                        print("→", end="")
                    elif best_action == 2:
                        print("↓", end="")
                    elif best_action == 3:
                        print("←", end="")
            print("")
        print("")


if "__main__" == __name__:
    # env = GridWorld(winds=[(2, 2, 3, 0.3)], step_cost=-0.1)
    env = GridWorld(
        size=(5, 4),
        start=(0, 0),
        goal=(4, 0),
        walls=[()],
        winds=[(3, 0, 3, 0.99), (3, 1, 3, 0.99), (3, 2, 3, 0.99)],
        step_cost=-0.05,
    )
    # print("\n" + "Initial state, representation of the environment")
    # print("S: start, G: goal, W: wall, ^>v<: wind direction \n")
    # env.render()

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

    env.first_visit_mc_control(num_iterations=1000, epsilon=0.1, plot="iteration")
