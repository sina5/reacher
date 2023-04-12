import numpy as np


class OUNoise:
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        np.random.seed(seed)

        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (
            self.mu - x
        ) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


class NormalNoise:
    def __init__(self, size, seed):
        np.random.seed(seed)
        self.size = size

    def reset(self):
        pass

    def sample(self):
        return np.random.normal(0, 0.1, size=self.size)
