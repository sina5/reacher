from distutils.version import StrictVersion

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dmodels import Actor, Critic
from noise import NormalNoise, OUNoise
from rbuffer import ReplayBuffer

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif (
    StrictVersion(torch.__version__) >= StrictVersion("1.13")
    and torch.backends.mps.is_available()
    and torch.backends.mps.is_built()
):
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class DDPGAgent:
    def __init__(
        self,
        state_size,
        action_size,
        buffer_size=int(1e6),
        batch_size=1024,
        gamma=0.99,
        tau=1e-3,
        lr_actor=1e-4,
        lr_critic=1e-3,
        weight_decay=0,
        noise_type="normal",
        # update_every=25,
        # num_experiences=15,
        seed=0,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size)
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
        # self.update_every = update_every
        # self.num_experiences = num_experiences
        if noise_type == "normal":
            self.noise = NormalNoise(action_size, seed)
        elif noise_type == "ou":
            self.noise = OUNoise(action_size, seed)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
            # np.random.normal(0, 0.1, size=self.action_size)
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def step(self, states, actions, rewards, next_states, dones, episode):
        for state, action, reward, next_state, done in zip(
            states, actions, rewards, next_states, dones
        ):
            self.memory.add(state, action, reward, next_state, done)

        if (
            len(self.memory) > self.batch_size
        ):  # and episode % self.update_every:
            # for _ in range(self.num_experiences):
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Update critic
        Q_targets_next = self.critic_target(
            next_states, self.actor_target(next_states)
        )
        # Q_targets = rewards.reshape(-1, 1) + (
        #     self.gamma
        #     * Q_targets_next.reshape(-1, 1)
        #     * (1 - dones.reshape(-1, 1))
        # )
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
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
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
