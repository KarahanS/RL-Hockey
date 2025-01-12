from typing import Iterable

import numpy as np

from .DDQN import DDQNAgent


class Round:
    """
    A class to represent a sequence of opponents to train against
    """

    def __init__(self, max_ep, agent_opp):
        self.max_ep = max_ep
        self.agent_opp = agent_opp


class Stats:
    """
    A class to represent the statistics of the training process
    """

    def __init__(self, returns=[], returns_ts=[], losses=[], losses_ts=[]):
        self.returns = returns
        self.returns_training_stages = returns_ts
        self.losses = losses
        self.losses_training_stages = losses_ts


def train_ddqn_agent(agent: DDQNAgent, env, max_steps: int, rounds: Iterable[Round],
                stats: Stats, ddqn_iter_fit=32, print_freq=25, tqdm=None, verbose=False):
    """
    Train the agent in the hockey environment

    Parameters:
    agent: the agent to train
    env: the environment to train in (should be a HockeyEnv object, can't import it for type hint...)
    max_steps: the maximum number of steps to train for each episode
    rounds: describing the sequence of opponents to train against
    stats: object to store the statistics of the training process
    ddqn_iter_fit: the number of iterations to train the DDQN agent for
    tqdm: tqdm object (optional, for differentiating between notebook and console)
    print_freq: how often to print the current episode statistics
    """
    
    # TODO: env.reset supports changing modes, support different modes in this function. Randomly select mode?

    for j, r in enumerate(rounds):
        max_ep = r.max_ep
        agent_opp = r.agent_opp
        if verbose:
            print(f"Begin round {j+1} for {max_ep} episodes")

        stats.losses_training_stages.append(len(stats.losses))
        stats.returns_training_stages.append(len(stats.returns))

        if tqdm is None:
            tqdm = lambda x: x

        for i in tqdm(range(max_ep)):
            total_reward = 0
            ob_a1, _info = env.reset()
            ob_a2 = env.obs_agent_two()

            for t in range(max_steps):
                done = False
                trunc = False

                a1_discr = agent.act(ob_a1)
                a1 = env.discrete_to_continous_action(a1_discr)
                a2 = agent_opp.act(ob_a2)

                ob_a1_next, reward, done, trunc, _info = env.step(
                    np.hstack([a1, a2])
                )
                total_reward += reward

                agent.store_transition(
                    (ob_a1, a1_discr, reward, ob_a1_next, done)
                )

                ob_a1 = ob_a1_next
                ob_a2 = env.obs_agent_two()

                if done or trunc:
                    break
            
            loss = agent.train(ddqn_iter_fit)

            stats.losses.extend(loss)
            stats.returns.append([i, total_reward, t+1])

            if verbose and (i % print_freq == 0 or i == max_ep - 1):
                print(f"Episode {i+1} | Return: {total_reward} | Loss: {loss[-1]} | Done in {t+1} steps")
