"""Implementation of Monte Carlo Control algorithm for the gridworld."""

import pickle
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class MCControl:
    """Monte Carlo Control algorithm for the gridworld."""

    def __init__(
        self,
        env,
        num_episodes=100,
        epsilon=0.1,
        gamma=0.9,
        plot="episode",
        load=None,
        save="mccontrol.pkl",
    ):
        self.env = env
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.gamma = gamma
        self.plot = plot
        self.load = load
        self.save = save

    def run(self) -> None:
        """Perform on-policy first-visit Monte Carlo control, using epsilon-greedy exploration"""
        if not self.load:
            # Initialize an arbitrary epsilon-soft policy
            policy = {}
            for state in self.env.states:
                for action in self.env.action_space():
                    policy[(state, action)] = 1 / len(self.env.action_space())
            # Initialize the state-action value function to arbitrary values for each state-action pair
            q = {}
            for state in self.env.states:
                for action in self.env.action_space():
                    q[(state, action)] = 0
        else:
            pickle_in = open(self.load, "rb")
            policy, q = pickle.load(pickle_in)

        # Empty list of returns for each state-action pair
        returns = defaultdict(list)

        # For the second task
        q_list = np.zeros(self.num_episodes)

        # Initialize an empty list to store the average return per episode
        average_returns = []

        # Initialize an empty list to store the time per episode
        episode_times = []

        # Store start time
        start_time = time.time()

        # For each iteration...
        for i in tqdm(range(self.num_episodes)):
            # Generate an episode using the epsilon-soft policy
            self.env.reset()
            episode = []
            while True and len(episode) < 1000:
                # Choose an action from the action space, based on its probability in the policy
                action = np.random.choice(
                    self.env.action_space(),
                    p=[policy[(self.env.state, a)] for a in self.env.action_space()],
                )
                next_state = self.env.state_transition(self.env.state, action)
                reward = self.env.reward(next_state)
                episode.append((self.env.state, action, reward))
                self.env.state = next_state
                if self.env.state == self.env.goal:
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
                    G = sum([x[2] * self.gamma**i for i, x in enumerate(episode[j:])])
                    # Add the concrete return to the list of returns for the state
                    returns[(state, action)].append(G)
                    # Calculate the mean of all returns for the state and update the value function
                    q[(state, action)] = np.mean(returns[(state, action)])

            # Update the policy to be greedy with respect to the state-action value function
            for state, _, _ in episode:
                # Select action greedily with respect to the state-action value function
                best_action = np.argmax(
                    [q[(state, action)] for action in self.env.action_space()]
                )

                # Update the policy (epsilon-soft)
                for action in self.env.action_space():
                    if action == self.env.action_space()[best_action]:
                        policy[(state, action)] = (
                            1
                            - self.epsilon
                            + self.epsilon / len(self.env.action_space())
                        )
                    else:
                        policy[(state, action)] = self.epsilon / len(
                            self.env.action_space()
                        )

            # Store the average return of this episode
            average_return = sum([x[2] for x in episode]) / len(episode)
            average_returns.append(average_return)

            # Store the time of this episode
            episode_times.append(time.time() - start_time)

            # Store Q-value of very first state-action pair
            q_list[i] = q[((0, 0), 0)]

        # print(
        #     f"Training finished after {self.num_episodes} iterations. Resulting policy:"
        # )
        # self.env.print_policy(policy)
        if self.plot:
            if self.plot == "episode":
                # Plot the returns obtained in each episode
                plt.plot(range(self.num_episodes), average_returns)
                plt.xlabel("Episode")
            elif self.plot == "wallclock":
                # Plot the returns obtained in each episode
                plt.plot(episode_times, average_returns)
                plt.xlabel("Time (seconds)")
            plt.ylabel("Mean return-per-episode")
            plt.show()

        if self.save:
            # Save policy and q
            file = open(self.save, "wb")
            pickle.dump((policy, q), file)
            file.close()

        return q_list
