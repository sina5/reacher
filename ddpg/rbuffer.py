import random
from collections import deque, namedtuple
from distutils.version import StrictVersion

import numpy as np
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Choose GPU if available
elif (
    StrictVersion(torch.__version__) >= StrictVersion("1.13")
    and torch.backends.mps.is_available()
    and torch.backends.mps.is_built()
):
    device = torch.device("mps")  # Choose Apple GPU if available
else:
    device = torch.device("cpu")  # If no GPUs, use CPU


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        """A memory of different actions and environment trajectories.

        Args:

            buffer_size (int): Number of trajectories to store in memory
            batch_size (int): Size of minibatches
        """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def add(self, state, action, reward, next_state, done):
        """Add a trajectory to replay buffer based on current state.

        Args:
            state (float): Current state
            action (float): Action taken
            reward (float): Reward of the action in this state
            next_state (float): Next state
            done (bool): if episodes has finished
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Get a sample trajectory from memory

        Returns:
            tuple: A tuple of states, actions, rewards, next_states, and dones.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(
                np.vstack([e.state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.vstack(
                    [e.done for e in experiences if e is not None]
                ).astype(np.uint8)
            )
            .float()
            .to(device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
