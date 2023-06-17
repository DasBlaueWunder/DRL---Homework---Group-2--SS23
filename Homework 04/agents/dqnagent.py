"""DQN agent class for training and testing."""

import random

import time
from easydict import EasyDict

import wandb
import gymnasium as gym

import torch
import torch.nn.functional as F
from torch.backends import cudnn

from tqdm import tqdm
import numpy as np

from graphs.models.dqn import DQN
from utils.replay_buffer import ReplayBuffer, Transition

cudnn.benchmark = True

# pylint: disable=no-member
# pylint: disable=not-callable


class DQNAgent:
    """DQN agent class for training and testing.

    Args:
        config: EasyDict containing configuration parameters.
        test: Boolean indicating whether to run in test mode.

    Attributes:
        config (EasyDict): EasyDict containing configuration parameters.
        test (bool): Boolean indicating whether to run in test mode.
        checkpoint_file (str): Path to checkpoint file, None if no checkpoint.
        cuda (bool): Boolean indicating whether to use CUDA.
        device (torch.device): Device to use for computations.
        env (gym.Env): Gym environment.
        last_observation (np.ndarray): Last observation from environment.
        model (DQN): DQN model.
        target (DQN): Target network.
        replay_buffer (ReplayBuffer): Replay buffer.
        steps_since_train (int): Number of steps since last training step.
        epochs_since_target_update (int): Number of epochs since last target update.
        step_num (int): Number of iterations.
        episode_rewards (list): List of episode rewards.
        rolling_reward (int): Rolling reward.
        eps (float): Current epsilon value.
    """

    def __init__(self, config: EasyDict, test):
        # load config
        self.config = config

        # enable/disable test mode
        self.test = test

        # load checkpoint
        try:
            self.checkpoint_file = config.checkpoint_file
        except AttributeError:
            self.checkpoint_file = None

        # enable/disable cuda
        if self.config.cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # load environment
        if self.test:
            self.env = gym.make(config.env_name, render_mode="human")
        else:
            self.env = gym.make(config.env_name)
        self.last_observation, _ = self.env.reset()

        # load model
        self.model = DQN(
            self.env.observation_space.shape, self.env.action_space.n, config.lr
        ).to(self.device)
        if self.checkpoint_file is not None:
            print(f"Loading checkpoint from {self.checkpoint_file}")
            self.model.load_state_dict(torch.load(self.checkpoint_file))
        else:
            print("No checkpoint file specified, starting from scratch")

        # load target network
        self.target = DQN(
            self.env.observation_space.shape, self.env.action_space.n, config.lr
        ).to(self.device)
        self.update_target_network()

        self.replay_buffer = ReplayBuffer(config.memory_capacity)
        self.steps_since_train = 0
        self.epochs_since_target_update = 0

        self.step_num = 0

        self.episode_rewards = []
        self.rolling_reward = 0

        self.eps = config.eps_start

    def get_action(self):
        """Get action from model using epsilon-greedy policy."""
        self.eps = max(self.config.eps_end, self.config.eps_decay**self.step_num)
        if self.test:
            self.eps = 0
        if random.random() < self.eps:
            action = self.env.action_space.sample()
        else:
            action = (
                self.model(torch.Tensor(self.last_observation).to(self.device))
                .max(-1)[-1]
                .item()
            )
        return action

    def update_target_network(self):
        """Update target network with model weights."""
        self.target.load_state_dict(self.model.state_dict())

    def train_step(self):
        """Perform one training step."""
        # sample from replay buffer
        state_transitions = self.replay_buffer.sample(self.config.batch_size)

        # convert the batch to tensors
        cur_states = torch.stack(([torch.Tensor(s[0]) for s in state_transitions])).to(
            self.device
        )
        rewards = torch.stack(([torch.Tensor([s[3]]) for s in state_transitions])).to(
            self.device
        )
        mask = torch.stack(
            (
                [
                    torch.Tensor([0]) if s[4] else torch.Tensor([1])
                    for s in state_transitions
                ]
            )
        ).to(self.device)
        next_states = torch.stack(([torch.Tensor(s[2]) for s in state_transitions])).to(
            self.device
        )
        actions = [s[1] for s in state_transitions]

        # compute next q values
        with torch.no_grad():
            qvals_next = self.target(next_states).max(-1)[0]

        self.model.opt.zero_grad()
        qvals = self.model(cur_states)
        one_hot_actions = F.one_hot(
            torch.LongTensor(actions), self.env.action_space.n
        ).to(self.device)

        # TODO: replace with Huber loss
        loss = (
            (
                rewards
                + mask[:, 0] * self.config.gamma * qvals_next
                - torch.sum(qvals * one_hot_actions, -1)
            )
            ** 2
        ).mean()
        loss.backward()
        self.model.opt.step()
        return loss

    def train(self):
        """Outer training loop. Performs training steps until done."""
        progress = tqdm()
        try:
            while True:
                if self.test:
                    self.env.render()
                    time.sleep(0.05)
                progress.update(1)
                action = self.get_action()
                observation, reward, done, truncated, _ = self.env.step(action)
                self.rolling_reward += reward

                reward = reward * 0.1
                self.replay_buffer.insert(
                    Transition(self.last_observation, action, observation, reward, done)
                )
                self.last_observation = observation

                if done or truncated:
                    self.episode_rewards.append(self.rolling_reward)
                    if self.test:
                        print(f"Episode reward: {self.rolling_reward}")
                    self.rolling_reward = 0
                    self.last_observation, _ = self.env.reset()

                self.steps_since_train += 1
                self.step_num += 1

                if (
                    (not self.test)
                    and self.replay_buffer.index > self.config.batch_size
                    and self.steps_since_train > self.config.train_frequency
                ):
                    loss = self.train_step()

                    wandb.log(
                        {
                            "loss": loss.detach().cpu().item(),
                            "epsilon": self.eps,
                            "avg_reward": np.mean(self.episode_rewards),
                        },
                        step=self.step_num,
                    )
                    self.episode_rewards = []
                    self.epochs_since_target_update += 1
                    if self.epochs_since_target_update > self.config.target_update:
                        print("Updating target network")
                        self.update_target_network()
                        self.epochs_since_target_update = 0
                        torch.save(
                            self.target.state_dict(),
                            f"{self.config.model_dir}/{self.step_num}.pth",
                        )

                    self.steps_since_train = 0

        except KeyboardInterrupt:
            pass
        finally:
            self.env.close()
