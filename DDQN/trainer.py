import copy
import os
import sys
from enum import Enum
from typing import Iterable
from pathlib import Path
from threading import Thread, Lock

import numpy as np
import torch
import wandb

root_dir = os.path.dirname(os.path.abspath("../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from DDQN.evaluation import compare_agents, display_stats
from DDQN.DQN import DQNAgent
from hockey.hockey_env import Mode as HockeyMode
from hockey.hockey_env import HockeyEnv, BasicOpponent


class CustomHockeyMode(Enum):
    """Extension of the HockeyEnv Mode class"""

    NORMAL = 0
    SHOOTING = 1
    DEFENSE = 2
    RANDOM_SHOOTING_DEFENSE = 3
    RANDOM_ALL = 4

    def __str__(self):
        return self.name


class RandomWeaknessBasicOpponent(BasicOpponent):
    def __init__(self, weakness_prob: float = 0.5):
        super().__init__()
        self.weakness_prob = weakness_prob
    
    def update_weakness(self):
        if np.random.rand() < self.weakness_prob:
            self.weak = True
        else:
            self.weak = False


class Round:
    """
    A class to represent a sequence of opponents to train against
    """

    def __init__(self, max_ep: int, agent_opp: DQNAgent | BasicOpponent, game_mode: CustomHockeyMode,
                 train_opp: bool = False):
        self.max_ep = max_ep
        self.agent_opp = agent_opp
        self.game_mode = game_mode
        self.train_opp = train_opp

        if train_opp:
            assert isinstance(agent_opp, DQNAgent), "Opponent must be a DQNAgent to train it"
    
    def __str__(self):
        return f"Round(max_ep={self.max_ep}, agent_opp={str(type(self.agent_opp))[8:-2]}, " \
            f"game_mode={self.game_mode}" + (", train_opp" if self.train_opp else "") + ")"


class Stats:
    """
    A class to represent the statistics of the training process
    """

    def __init__(self, returns=[], returns_ts=[], losses=[], losses_ts=[]):
        self.returns = returns
        self.returns_training_stages = returns_ts
        self.losses = losses
        self.losses_training_stages = losses_ts


def eval_task(agent_copy: DQNAgent, opps_dict_copy: dict, env_copy: HockeyEnv, curr_ep: int,
              curr_round_ep: int, max_eps: int, eval_num_matches: int, wandb_hparams: dict,
              print_lock: Lock, verbose=False):
    def eval_opp(agent_loc: DQNAgent, name_loc: str, opp_loc: DQNAgent | BasicOpponent,
                 env_loc: HockeyEnv):
        comp_stats = compare_agents(
            agent_loc, opp_loc, env_loc, num_matches=eval_num_matches
        )

        win_rate_player = np.mean(comp_stats["winners"] == 1)
        win_rate_opp = np.mean(comp_stats["winners"] == -1)
        draw_rate = np.mean(comp_stats["winners"] == 0)

        win_status_mean = np.mean(comp_stats["winners"])
        win_status_std = np.std(comp_stats["winners"])

        returns_player = np.sum(comp_stats["rewards_player"])
        returns_opp = np.sum(comp_stats["rewards_opp"])
        returns_diff = np.abs(returns_player - returns_opp)

        name_loc = name_loc.replace(" ", "_").lower()
        # Log the statistics to wandb
        if wandb_hparams is not None:
            wandb.log({
                "episode": curr_ep,
                f"eval/{name_loc}_player_win_rate": win_rate_player,
                f"eval/{name_loc}_opp_win_rate": win_rate_opp,
                f"eval/{name_loc}_draw_rate": draw_rate,
                f"eval/{name_loc}_returns_diff": returns_diff,
                f"eval/{name_loc}_win_status_mean": win_status_mean,
                f"eval/{name_loc}_win_status_std": win_status_std
            })
    
        if curr_round_ep == max_eps - 1 and verbose:
            with print_lock:
                print(f"Evaluated against opponent: {name}")
                display_stats(comp_stats, name, verbose=True)
    
    env_copy.mode(HockeyMode.NORMAL)
    for name, opp in opps_dict_copy.items():
        eval_opp(name, opp, env_copy)
    
    env_copy.mode(HockeyMode.TRAIN_SHOOTING)
    eval_opp("Shooting Mode", opps_dict_copy[0], env_copy)  # Opponent does not matter
    env_copy.mode(HockeyMode.TRAIN_DEFENSE)
    eval_opp("Defense Mode", opps_dict_copy[0], env_copy)  # Opponent does not matter

    del env_copy        
    del agent_copy
    del opps_dict_copy


def train_ddqn_agent_torch(agent: DQNAgent, env: HockeyEnv, model_dir: str, max_steps: int,
                rounds: Iterable[Round], stats: Stats, eval_opps_dict: dict, ddqn_iter_fit=32,
                eval_freq=500, eval_num_matches=1000, print_freq=25, tqdm=None, verbose=False,
                wandb_hparams=None):
    """
    Train the agent in the hockey environment

    Parameters:
    agent: the agent to train
    env: the environment to train in
    max_steps: the maximum number of steps to train for each episode
    rounds: describing the sequence of opponents to train against
    stats: object to store the statistics of the training process
    eval_opps_dict: dictionary containing the opponents to evaluate against and their names
    ddqn_iter_fit: the number of iterations to train the DDQN agent for
    eval_freq: the episode frequency of evaluating the agent
    eval_num_matches: the number of matches to evaluate the agent for
    tqdm: tqdm object (optional, for differentiating between notebook and console)
    print_freq: how often to print the current episode statistics
    wandb_hparams: hyperparameters to log to wandb
    """

    train_device = agent.train_device
    def np2gpu(data: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(data).float().to(train_device)
    
    total_eps = 0
    eval_threads = []
    print_lock = Lock()

    if wandb_hparams is not None:
        wandb_hparams["rounds"] = [str(r) for r in rounds]
        run_name = wandb_hparams.pop("run_name")

        # Initialize wandb
        run = wandb.init(
            # set the wandb project where this run will be logged
            project="RL-DDQN",
            name=run_name,

            # track hyperparameters and run metadata
            config=wandb_hparams
        )
        run_id = run.id
    else:
        run_id = None

    for j, r in enumerate(rounds):
        max_ep = r.max_ep
        agent_opp = r.agent_opp
        train_opp = r.train_opp

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
            print(f"Begin round {j+1} with mode {custom_mode} for {max_ep} episodes")

        stats.losses_training_stages.append(len(stats.losses))
        stats.returns_training_stages.append(len(stats.returns))

        if tqdm is None:
            tqdm = lambda x: x

        for i in tqdm(range(max_ep)):
            total_reward = 0
            if train_opp:
                opp_total_reward = 0
            
            ob_a1, _info = env.reset(mode=mode)
            ob_a2 = env.obs_agent_two()

            if isinstance(agent_opp, RandomWeaknessBasicOpponent):
                agent_opp.update_weakness()

            for t in range(max_steps):
                done = False
                trunc = False

                act_a1_discr = agent.act_torch(np2gpu(ob_a1))  # int
                act_a1 = env.discrete_to_continous_action(act_a1_discr)  # numpy array
                act_a2_raw = agent_opp.act(ob_a2)  # numpy array
                if isinstance(agent_opp, DQNAgent):
                    act_a2_discr = act_a2_raw
                    act_a2 = env.discrete_to_continous_action(act_a2_discr)
                else:
                    act_a2 = act_a2_raw

                ob_a1_next, reward, done, trunc, _info = env.step(
                    np.hstack([act_a1, act_a2])
                )
                total_reward += reward

                agent.store_transition(
                    (ob_a1, act_a1_discr, reward, ob_a1_next, done)
                )

                ob_a2_next = env.obs_agent_two()

                if train_opp:
                    opp_info = env.get_info_agent_two()
                    opp_reward = env.get_reward_agent_two(opp_info)
                    opp_total_reward += opp_reward

                    agent_opp.store_transition(
                        (ob_a2, act_a2_discr, opp_reward, ob_a2_next, done)
                    )
                
                ob_a1 = ob_a1_next
                ob_a2 = ob_a2_next
                
                if done or trunc:
                    break
            
            fit_loss = agent.train_torch(ddqn_iter_fit)
            stats.losses.extend(fit_loss)
            stats.returns.append([i, total_reward, t+1])

            if train_opp:
                opp_fit_loss = agent_opp.train_torch(ddqn_iter_fit)

            # TODO: Reward could be utilized differently in each step.
            #   Logging here would have to be changed accordingly
            if wandb_hparams is not None:
                log_dict = {
                    "episode": total_eps,
                    "return": total_reward,
                    "loss": fit_loss[-1],
                    "steps": t+1
                }
                
                if train_opp:
                    log_dict["opp_return"] = opp_total_reward
                    log_dict["opp_loss"] = opp_fit_loss[-1]
                
                wandb.log(log_dict)

            if verbose and (i % print_freq == 0 or i == max_ep - 1):
                print(
                    f"Episode {i+1} | Return: {total_reward} | Loss: {fit_loss[-1]} | Done in {t+1} steps"
                )
            
            if (eval_freq > 0) and (total_eps % eval_freq == 0 or i == max_ep - 1):
                # Agent back-up
                for p in Path(model_dir).glob("Q_model_ep_*.ckpt"):
                    p.unlink()
                agent.save_state(model_dir, f"Q_model_ep_{total_eps}.ckpt")

                # Evaluate the agent
                env_copy = copy.deepcopy(env)
                agent_copy = copy.deepcopy(agent)
                eval_opps_dict_copy = copy.deepcopy(eval_opps_dict)
                eval_opps_dict_copy["Self-copied"] = agent_copy

                eval_thread = Thread(
                    target=eval_task,
                    args=(
                        agent_copy, eval_opps_dict_copy, env_copy, total_eps, i, max_ep,
                        eval_num_matches, wandb_hparams, print_lock, verbose
                    )
                )
                eval_thread.start()
                eval_threads.append(eval_thread)
            
            total_eps += 1
    
    for thread in eval_threads:
        thread.join()

    return run_id
