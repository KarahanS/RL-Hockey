import os
import sys
from enum import Enum
from typing import Iterable

import numpy as np
import torch

root_dir = os.path.dirname(os.path.abspath("../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from DDQN.DQN import DQNAgent
from hockey.hockey_env import Mode as HockeyMode
from hockey.hockey_env import HockeyEnv


class CustomHockeyMode(Enum):
    """Extension of the HockeyEnv Mode class"""

    NORMAL = 0
    SHOOTING = 1
    DEFENSE = 2
    RANDOM_SHOOTING_DEFENSE = 3
    RANDOM_ALL = 4

    def __str__(self):
        return self.name


class Round:
    """
    A class to represent a sequence of opponents to train against
    """

    def __init__(self, max_ep: int, agent_opp: DQNAgent, game_mode: CustomHockeyMode):
        self.max_ep = max_ep
        self.agent_opp = agent_opp
        self.game_mode = game_mode


class Stats:
    """
    A class to represent the statistics of the training process
    """

    def __init__(self, returns=[], returns_ts=[], losses=[], losses_ts=[]):
        self.returns = returns
        self.returns_training_stages = returns_ts
        self.losses = losses
        self.losses_training_stages = losses_ts


def train_ddqn_agent_gpu(agent: DQNAgent, env: HockeyEnv, max_steps: int, rounds: Iterable[Round],
                stats: Stats, ddqn_iter_fit=32, print_freq=25, tqdm=None, verbose=False):
    """
    Train the agent in the hockey environment

    Parameters:
    agent: the agent to train
    env: the environment to train in
    max_steps: the maximum number of steps to train for each episode
    rounds: describing the sequence of opponents to train against
    stats: object to store the statistics of the training process
    ddqn_iter_fit: the number of iterations to train the DDQN agent for
    tqdm: tqdm object (optional, for differentiating between notebook and console)
    print_freq: how often to print the current episode statistics
    """
    train_device = agent.train_device

    def np2gpu(data: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(data).float().to(train_device)

    for j, r in enumerate(rounds):
        max_ep = r.max_ep
        agent_opp = r.agent_opp

        custom_mode = r.game_mode
        if custom_mode == CustomHockeyMode.NORMAL:
            mode = HockeyMode.NORMAL
        elif custom_mode == CustomHockeyMode.SHOOTING:
            mode = HockeyMode.TRAIN_SHOOTING
        elif custom_mode == CustomHockeyMode.DEFENSE:
            mode = HockeyMode.TRAIN_DEFENSE
        elif custom_mode == CustomHockeyMode.RANDOM_SHOOTING_DEFENSE:
            mode = np.random.choice(
                [HockeyMode.TRAIN_SHOOTING, HockeyMode.TRAIN_DEFENSE]
            )
        elif custom_mode == CustomHockeyMode.RANDOM_ALL:
            mode = np.random.choice(
                [HockeyMode.NORMAL, HockeyMode.TRAIN_SHOOTING, HockeyMode.TRAIN_DEFENSE]
            )
        else:
            raise ValueError("Invalid mode")

        if verbose:
            print(f"Begin round {j+1} with mode {mode} for {max_ep} episodes")

        stats.losses_training_stages.append(len(stats.losses))
        stats.returns_training_stages.append(len(stats.returns))

        if tqdm is None:
            tqdm = lambda x: x

        for i in tqdm(range(max_ep)):
            total_reward = 0
            ob_a1, _info = env.reset(mode=mode)
            ob_a2 = env.obs_agent_two()

            for t in range(max_steps):
                done = False
                trunc = False

                act_a1_discr = agent.act_gpu(np2gpu(ob_a1))  # int
                act_a1 = env.discrete_to_continous_action(act_a1_discr)  # numpy array
                act_a2 = agent_opp.act(ob_a2)  # numpy array

                ob_a1_next, reward, done, trunc, _info = env.step(
                    np.hstack([act_a1, act_a2])
                )
                total_reward += reward

                agent.store_transition(
                    (ob_a1, act_a1_discr, reward, ob_a1_next, done)
                )

                ob_a1 = ob_a1_next
                ob_a2 = env.obs_agent_two()

                if done or trunc:
                    break
            
            fit_loss = agent.train_gpu(ddqn_iter_fit)
            stats.losses.extend(fit_loss)
            stats.returns.append([i, total_reward, t+1])

            if verbose and (i % print_freq == 0 or i == max_ep - 1):
                print(
                    f"Episode {i+1} | Return: {total_reward} | Loss: {fit_loss[-1]} | Done in {t+1} steps"
                )
    
    # Finished training: copy finalized Q network to CPU
    # TODO: May remove, see TODO in DQN.__init__
    agent.Q.load_state_dict(agent.Q_gpu.state_dict())


def train_ddqn_two_agents_gpu(agent_player: DQNAgent, agent_opp: DQNAgent, env: HockeyEnv, max_steps: int,
                rounds: Iterable[Round], stats: Stats, ddqn_iter_fit=32, print_freq=25, tqdm=None, verbose=False):
    """#TODO: docstring"""

    # TODO: implementation
    raise NotImplementedError

    return ...