import random
from collections import deque, namedtuple
from distutils.version import StrictVersion

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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


# Define actor network
class Actor(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        seed,
        fc1_units=256,
        fc2_units=128,
        fc3_units=64,
        fc4_units=32,
        fc5_units=16,
    ):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, fc4_units)
        self.fc5 = nn.Linear(fc4_units, fc5_units)
        self.fc6 = nn.Linear(fc5_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3, 3e-3)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)
        self.fc6.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return torch.tanh(self.fc6(x))


# Define critic network
class Critic(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        seed,
        fc1_units=256,
        fc2_units=128,
        fc3_units=64,
        fc4_units=32,
        fc5_units=16,
    ):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, fc4_units)
        self.fc5 = nn.Linear(fc4_units, fc5_units)
        self.fc6 = nn.Linear(fc5_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3, 3e-3)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.fc6(x)


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
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
        update_every=25,
        num_experiences=15,
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
        self.update_every = update_every
        self.num_experiences = num_experiences

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += np.random.normal(0, 0.1, size=self.action_size)
        return np.clip(action, -1, 1)

    def reset(self):
        pass

    def step(self, states, actions, rewards, next_states, dones, episode):
        for state, action, reward, next_state, done in zip(
            states, actions, rewards, next_states, dones
        ):
            self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size and episode % self.update_every:
            for _ in range(self.num_experiences):
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Update critic
        Q_targets_next = self.critic_target(
            next_states, self.actor_target(next_states)
        )
        Q_targets = rewards.reshape(-1, 1) + (
            self.gamma
            * Q_targets_next.reshape(-1, 1)
            * (1 - dones.reshape(-1, 1))
        )
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


def ddpg(
    agent,
    brain_name,
    env,
    n_episodes=500,
    max_t=1000,
    target_score=30,
    print_every=100,
):
    """DDPG Training.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    scores_window = deque(maxlen=100)  # last 100 scores
    checkpoint_path = None
    best_checkpoint_saved = False
    score_list = []
    avg_score_list = []
    first_score_match = 0
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        num_agents = len(env_info.agents)  # get number of agents
        scores = np.zeros(num_agents)  # initialize the score
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            done = env_info.local_done
            agent.step(states, actions, rewards, next_states, done, i_episode)
            states = next_states
            scores += rewards
            if np.any(done):
                break

        scores_window.append(scores)
        score_list.append(np.mean(scores))
        avg_score_list.append(np.mean(scores_window))

        # eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        if i_episode % print_every == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_window)
                )
            )
        if np.mean(scores_window) >= target_score:
            checkpoint_path = f"checkpoint_{i_episode}.pth"
            if not best_checkpoint_saved:
                print(
                    "\nEnvironment solved in {:d}"
                    " episodes!\tAverage Score: {:.2f}".format(
                        i_episode, np.mean(scores_window)
                    )
                )
                torch.save(
                    agent.critic_local.state_dict(),
                    f"critic_{checkpoint_path}",
                )
                torch.save(
                    agent.actor_local.state_dict(), f"actor_{checkpoint_path}"
                )
                print(f"Trained model weights saved to: {checkpoint_path}")
                best_checkpoint_saved = True
                first_score_match = i_episode
            break
        if i_episode == n_episodes:
            checkpoint_path = f"checkpoint_{i_episode}.pth"
            torch.save(
                agent.critic_local.state_dict(),
                f"critic_{checkpoint_path}",
            )
            torch.save(
                agent.actor_local.state_dict(), f"actor_{checkpoint_path}"
            )
            print(f"Trained model weights saved to: {checkpoint_path}")
    return score_list, avg_score_list, first_score_match
