"""Implementation of SARSA algorithm for the gridworld."""

import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class SARSA:
    """SARSA algorithm for the gridworld."""

    def __init__(
        self, env, num_episodes=100, gamma=1, alpha=0.1, epsilon=1, plot="episode"
    ):
        """Initialize SARSA."""
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.plot = plot
        self.Q = np.zeros((env.size[0], env.size[1], 4))
        self.Q[env.goal[0], env.goal[1], :] = num_episodes

    def choose_action(self, state):
        """Choose action based on epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(self.Q[state[0], state[1], :])
        return action

    def update(self, state, action, reward, next_state, next_action):
        """Update Q-values."""
        self.Q[state[0], state[1], action] += self.alpha * (
            reward
            + self.gamma * self.Q[next_state[0], next_state[1], next_action]
            - self.Q[state[0], state[1], action]
        )

    def run(self):
        """Run SARSA. Measure and plot average return per episode."""
        episode_time = np.zeros(self.num_episodes)
        returns = np.zeros(self.num_episodes)
        start_time = time.time()
        for episode in tqdm(range(self.num_episodes)):
            self.env.reset()
            total_return = 0
            state = self.env.state
            action = self.choose_action(state)
            while True:
                next_state = self.env.state_transition(state, action)
                reward = self.env.reward(next_state)
                if self.env.is_terminal_state(next_state):
                    break
                next_action = self.choose_action(next_state)
                self.update(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
                total_return += reward
            episode_time[episode] = time.time() - start_time
            # steps[episode] = step
            returns[episode] = total_return
            self.epsilon -= 2 / self.num_episodes if self.epsilon > 0.01 else 0.01
        # Plot average return per episode
        if self.plot == "episode":
            plt.plot(np.arange(1, self.num_episodes + 1), returns)
            plt.xlabel("Episode")
            plt.ylabel("Average Return")
            plt.show()
        elif self.plot == "wallclock":
            plt.plot(episode_time, returns)
            plt.xlabel("Wallclock Time")
            plt.ylabel("Average Return")
            plt.show()
        else:
            raise ValueError("plot should be 'episode' or 'iteration'")
