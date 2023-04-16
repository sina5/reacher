from collections import deque

import numpy as np
import torch


def ddpg(
    agent,
    brain_name,
    env,
    n_episodes=500,
    max_t=1000,
    target_score=30,
    print_every=100,
):
    """DDPG Model Training.
    Params
    ======
        agent (DDPGAgent): Training agent
        brain_name (str): Brain name to use in Unity env
        env (UnityEnvironment): Environment for agents
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        target_score (int): Target average score over 100 episodes
            to reach and stop training.
        print_every (int): interval to print average scores

    """

    scores_window = deque(maxlen=100)  # last 100 scores
    checkpoint_path = None
    best_checkpoint_saved = False
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
        avg_score_list.append(np.mean(scores_window))

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
    return avg_score_list, first_score_match
