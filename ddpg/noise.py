# based on https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum

import numpy as np


class OUNoise:
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Ornstein Uhlenbeck process noise
        Args:
            size (int): Output size of noise sampling
            seed (int): Random generator seed
            mu (float, optional): mean of process. Defaults to 0.0.
            theta (float, optional): long term mean of the process. Defaults to 0.15.
            sigma (float, optional): Standard deviation. Defaults to 0.2.
        """
        np.random.seed(seed)

        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        """Reset noise current state to its mean."""
        self.state = np.copy(self.mu)

    def sample(self):
        """Get a sample from noise distribution

        Returns:
            np.array[np.float32]: An array of random
                number sampled from noise distribution.
        """
        x = self.state
        dx = self.theta * (
            self.mu - x
        ) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


class NormalNoise:
    def __init__(self, size, seed):
        """Normal (Gaussian process) noise

        Args:
            size (int): Output size of noise sampling
            seed (int): Random generator seed
        """
        np.random.seed(seed)
        self.size = size

    def reset(self):
        pass

    def sample(self):
        """Get a sample from noise distribution

        Returns:
            np.array[np.float32]: An array of random
                number sampled from noise distribution.
        """
        return np.random.normal(0, 0.1, size=self.size)
