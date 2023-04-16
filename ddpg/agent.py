# based on https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum
from distutils.version import StrictVersion

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dmodels import Actor, Critic
from noise import NormalNoise, OUNoise
from rbuffer import ReplayBuffer

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


class DDPGAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        buffer_size: int = int(1e6),
        batch_size: int = 1024,
        gamma: float = 0.99,
        tau: float = 1e-3,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        weight_decay: float = 0,
        noise_type: str = "normal",
        update_every: int = 1,
        seed: int = 0,
    ):
        """
        DDPG agent interacts with a given environment and trains
        two neural networks to find optimal actions.

        Args:
            state_size (int):  each state dimension
            action_size (int): action space size
            buffer_size (int, optional): replay buffer size. Defaults to int(1e6).
            batch_size (int, optional):
                batch size to get trajectories from replay buffer. Defaults to 1024.
            gamma (float, optional): discount factor. Defaults to 0.99.
            tau (float, optional): soft update weights factor. Defaults to 1e-3.
            lr_actor (float, optional): actor net learning rate. Defaults to 1e-4.
            lr_critic (float, optional): critic net learning rate. Defaults to 1e-3.
            weight_decay (float, optional): weight decay coefficient. Defaults to 0.
            noise_type (str, optional):
                type of noise. Defaults to "normal". Choices are:
                - "normal" for normal noise
                - "ou" for Ornstein Uhlenbeck process noise
            update_every (int, optional):
                number of steps between each network update. Defaults to 1.
            seed (int, optional): random generator seed. Defaults to 0.
        """

        # Set default variables
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # Initialize a replay buffer as a memory of trajectories
        self.memory = ReplayBuffer(buffer_size, batch_size)

        # Assign DNNs for actor and critic
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=lr_actor
        )
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=lr_critic,
            weight_decay=weight_decay,
        )

        # Number of episode between network updates
        self.update_every = update_every
        torch.manual_seed(seed)

        # Choose noise type to add to actions
        if noise_type == "normal":
            # Normal (Gaussian process) noise
            self.noise = NormalNoise(action_size, seed)
        elif noise_type == "ou":
            # Ornstein Uhlenbeck process noise
            self.noise = OUNoise(action_size, seed)

    def act(self, state, add_noise: bool = True):
        """Choose action to take based on current environment state
        and actor network output.

        Args:
            state (np.array): Array of floats. Environment current state.
            add_noise (bool, optional): Wether to add noise to actions. Defaults to True.

        Returns:
            np.array: Array of floats. An action to take for each agent
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        # disable model gradient updates
        with torch.no_grad():
            # get an array of actions from actor network output
            action = self.actor_local(state).cpu().data.numpy()
        # enable model training
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()

        # clip actions to be in the range of (-1,1)
        return np.clip(action, -1, 1)

    def reset(self):
        """Reset OUNoise state"""
        self.noise.reset()

    def step(self, states, actions, rewards, next_states, dones, episode):
        """Take a step in the environment and calculate rewards

        Args:
            states (np.array[np.float32]): Current agents' states
            actions (np.array[np.float32]): Actions taken in this state
            rewards (np.array[np.float32]): Rewards for current state
            next_states (np.array[np.float32]): Get an array of next states
            dones (np.array[np.bool]): If episodes are finished
            episode (int): Current episode index
        """
        for state, action, reward, next_state, done in zip(
            states, actions, rewards, next_states, dones
        ):
            self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size and episode % self.update_every:
            # Get a list of experiences from replay buffer
            experiences = self.memory.sample()
            # Update the models based on experiences
            self.learn(experiences)

    def learn(self, experiences):
        """Update actor and critic based on experiences

        Args:
            experiences (np.array[np.float32]):
                A list of experiences chosen from memory buffer
        """
        states, actions, rewards, next_states, dones = experiences

        # Update critic
        Q_targets_next = self.critic_target(
            next_states, self.actor_target(next_states)
        )
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Update target models based on local models

        Args:
            local_model (nn.Module): DNN to get weights from
            target_model (nn.Module): DNN to update based on local model
            tau (float): soft update weights factor
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
