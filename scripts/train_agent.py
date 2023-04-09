import sys

sys.path.insert(0, "../mlagents")
sys.path.insert(1, "../ddpg")

import os
from argparse import ArgumentParser
from typing import List

import numpy as np
import plotly.graph_objects as go
from unityagents import UnityEnvironment

from ddpg import DDPGAgent, ddpg


def parse_args():
    parser = ArgumentParser(
        prog="Reacher Agent Training",
        description="Trains an RL agent to reach",
    )
    parser.add_argument(
        "-u",
        "--unity-app",
        type=str,
        help="Path to a reacher unity app",
        default="../Reacher.app",
    )
    parser.add_argument(
        "-t",
        "--target-score",
        type=int,
        help="Target training score",
        default=30,
    )
    return vars(parser.parse_args())


def start_unity_env(file_name, worker_id=10):
    env = UnityEnvironment(file_name=file_name, worker_id=worker_id)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    print(env.brain_names)
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents in the environment
    print("Number of agents:", len(env_info.agents))
    # number of actions
    action_size = brain.vector_action_space_size
    print("Number of actions:", action_size)
    # examine the state space
    state = env_info.vector_observations[0]
    state_size = len(state)
    print("States have length:", state_size)
    return env, brain_name, state_size, action_size


def plot_scores(
    scores: List[float],
    first_score_match: int = None,
    target_score: int = 13,
    output_file: str = "../images/train_scores.png",
):
    fig = go.Figure(
        data=go.Scatter(x=np.arange(len(scores)), y=scores, name="Scores")
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(scores)),
            y=target_score * np.ones(len(scores)),
            name="Target",
        )
    )

    fig.update_layout(
        annotations=[
            go.layout.Annotation(
                x=first_score_match,
                y=target_score,
                showarrow=True,
                arrowhead=4,
                arrowwidth=2,
                ax=first_score_match - 10,
                ay=target_score + 4,
                text="Agent passed score threshold here",
            )
        ]
    )

    fig.update_layout(
        title="Scores",
        xaxis_title="Episodes",
        yaxis_title="Score",
    )
    fig.update_layout(
        font={
            "family": "Nunito",
            "size": 12,
            "color": "#707070",
        },
        title={
            "font": {
                "family": "Lato",
                "size": 14,
                "color": "#1f1f1f",
            },
        },
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )

    fig.write_image(output_file)


if __name__ == "__main__":
    args = parse_args()

    ua = args.get("unity_app", "../Reacher.app")
    if os.path.exists(ua):
        env, brain_name, state_size, action_size = start_unity_env(ua)
    else:
        raise FileNotFoundError

    agent = DDPGAgent(
        state_size=state_size,
        action_size=action_size,
        seed=0,
    )
    target_score = args.get("target_score", 30)
    score_list, avg_score_list, first_score_match = ddpg(
        agent,
        brain_name,
        env,
        n_episodes=200,
        target_score=target_score,
        print_every=5,
    )
    plot_scores(avg_score_list, first_score_match, target_score)
    env.close()
