import os
import sys

import numpy as np

root_dir = os.path.dirname(os.path.abspath("./"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from DDQN.DQN import DQNAgent
from hockey.hockey_env import HockeyEnv, BasicOpponent


def compare_agents(agent_player: DQNAgent, agent_opp: DQNAgent | BasicOpponent, env: HockeyEnv,
                   num_matches=100, render=False, tqdm=None, seed=42):
    """
    Play a number of matches between two agents, display and return statistics

    Parameters:
    agent_player: the agent to play as the player
    agent_opponent: the agent to play as the opponent
    env: the environment to train in
    num_matches: the number of matches to play
    render: whether to render the environment
    tqdm: tqdm object (optional, for differentiating between notebook and console)

    Returns:
    win_rate: the win rate of the player agent
    """

    stats = {
        "winners": [],
        "rewards_player": [],
        "rewards_opp": [],
        "obs_player": [],
        "obs_opp": []
    }

    np.random.seed(seed)
    try:
        env.set_seed(seed)
    except AttributeError:
        env.seed(seed)

    if tqdm is None:
        tqdm = lambda x: x

    for _ in tqdm(range(num_matches)):
        done = False
        trunc = False

        obs, _ = env.reset()
        obs_opp = env.obs_agent_two()
        while not (done or trunc):
            if render:
                env.render()
            
            a1_discr = agent_player.act(obs)
            a1 = env.discrete_to_continous_action(a1_discr)
            a2 = agent_opp.act(obs_opp)

            obs, reward, done, trunc, info_player = env.step(np.hstack([a1, a2]))
            info_opp = env.get_info_agent_two()
            reward_opp = env.get_reward_agent_two(info_opp)
            obs_opp = env.obs_agent_two()

            stats["obs_player"].append(obs)
            stats["obs_opp"].append(obs_opp)
            stats["rewards_player"].append(reward)
            stats["rewards_opp"].append(reward_opp)

            if done or trunc:
                stats["winners"].append(info_player["winner"])
                break

    stats_np = {k: np.asarray(v) for k, v in stats.items()}

    return stats_np


def display_stats(stats_np, verbose=False):
    """
    Display statistics from compare_agents

    Parameters:
    stats_np: the statistics to display with numpy arrays
    """
    
    def print_observation_stats(observation):
        print("  x pos player one:", np.mean(observation[0]))
        print("  y pos player one:", np.mean(observation[1]))
        print("  angle player one:", np.mean(observation[2]))
        print("  x vel player one:", np.mean(observation[3]))
        print("  y vel player one:", np.mean(observation[4]))
        print("  angular vel player one:", np.mean(observation[5]))
        print("  x player two:", np.mean(observation[6]))
        print("  y player two:", np.mean(observation[7]))
        print("  angle player two:", np.mean(observation[8]))
        print("  y vel player two:", np.mean(observation[9]))
        print("  y vel player two:", np.mean(observation[10]))
        print("  angular vel player two:", np.mean(observation[11]))
        print("  x pos puck:", np.mean(observation[12]))
        print("  y pos puck:", np.mean(observation[13]))
        print("  x vel puck:", np.mean(observation[14]))
        print("  y vel puck:", np.mean(observation[15]))
        print("  left player puck keep time:", np.mean(observation[16]))
        print("  right player puck keep time:", np.mean(observation[17]))

    if verbose:
        print("Player Observation Mean:")
        print_observation_stats(np.mean(stats_np["obs_player"], axis=0))
        print()

        print("Relative Std. Change in Agent Observations:")
        print_observation_stats(
            (np.std(stats_np["obs_player"], axis=0) - np.std(stats_np["obs_opp"], axis=0)) \
                / np.std(stats_np["obs_player"], axis=0)
        )
        print()
    
    print("Player Win Rate:", np.mean(stats_np["winners"] == 1))
    print("Opponent Win Rate:", np.mean(stats_np["winners"] == -1))
    print("Draw Rate:", np.mean(stats_np["winners"] == 0))
    print()

    print("Win Status (1 for win, 0 for draw, -1 for loss):")
    print("  Mean:", np.mean(stats_np["winners"]))
    print("  Std:", np.std(stats_np["winners"]))
    print()

    print("Returns:")
    print("  Player:", np.sum(stats_np["rewards_player"]))
    print("  Opponent:", np.sum(stats_np["rewards_opp"]))
    print("  Difference:",
        np.abs(np.sum(stats_np["rewards_player"]) - np.sum(stats_np["rewards_opp"]))
    )
