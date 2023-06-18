"""Implementation of a replay buffer"""

from random import sample
from collections import namedtuple

Transition = namedtuple(
    "Transition",
    ("last_observation", "action", "reward", "observation", "done"),
)


class ReplayBuffer(object):  # TODO: implement prioritized experience replay
    def __init__(self, capacity):
        self.buffer_size = capacity
        self.buffer = [None] * capacity
        self.index = 0

    def insert(self, transition):
        self.buffer[self.index % self.buffer_size] = transition
        self.index += 1

    def __len__(self):
        """Not the actual length but the number of transitions inserted so far."""
        return min(self.index, self.buffer_size)

    def sample(self, batch_size):
        assert batch_size < min(self.index, self.buffer_size)
        if self.index < self.buffer_size:
            return sample(self.buffer[: self.index], batch_size)
        return sample(self.buffer, batch_size)
