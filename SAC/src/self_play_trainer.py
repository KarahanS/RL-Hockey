#!/usr/bin/env python
"""
SAC Trainer for Hockey Environment with Custom Classes

This script trains an agent using Soft Actor-Critic (SAC) in the hockey
environment. It uses custom classes to wrap the hockey environment to a 
Gym-like API and provides support for opponent play via simple opponent classes.

*** SELF-PLAY MODIFICATIONS HIGHLIGHTED ***
"""
import sys
sys.path.append("../")  # Adjust if needed so that TD3, etc. are importable.


from DDQN.DDQN import DoubleDuelingDQNAgent
from DDQN.action_space import CustomActionSpace
from TD3.src.td3 import TD3  # your TD3 implementation
import argparse
import os
import torch
import numpy as np
import gymnasium as gym
import json
import pickle
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import concurrent.futures
import random  # <--- for random choice in self-play
import copy    # <--- for deep-copies of agent states

from sac import SACAgent
from memory import PrioritizedExperienceReplay
from hockey_env import HockeyEnv, BasicOpponent, Mode, BasicDefenseOpponent, BasicAttackOpponent
from noise import *  # your noise module
import collections

import os
from pathlib import Path




class PrioritizedOpponentBuffer():
    def __init__(self, B=1, xi=1, gamma=.95, tau=None):
        self.B = B
        if tau is None: self.tau = min(1e3, int(np.log(1e-2) / np.log(gamma)))
        else: self.tau = tau
        self.xi = 1
        self.gamma = gamma
        self.opponents = []
        self.history = collections.deque(self.tau * [-1], self.tau)
        self.history_outcomes = collections.deque(self.tau * [-1], self.tau)
        self.t = 0
        self.K = 0

    def add_opponent(self, opponent):
        self.opponents.append(opponent)
        self.K = len(self.opponents)
        self.t = min(self.t, self.tau)
        
    def get_opponent(self):
        if self.K < 1:
            print('The buffer is empty!')
            return None
        if self.t < self.K:
            opponent = self.opponents[self.t]
            return self.t, opponent
        else:
            opponent_history = (self.history == np.arange(self.K).reshape(-1, 1)).astype(int)
            discount = (self.gamma ** np.arange(self.tau))[::-1]
            N = np.sum(opponent_history * discount, axis=1)
            X = np.sum(opponent_history * self.history_outcomes * discount, axis=1) / N
            c = 2 * self.B * np.sqrt(self.xi * np.log(np.sum(N)) / N)
            final = np.nan_to_num(X + c, copy=False, nan=np.inf).flatten()
            opponent_idx = np.argmax(final)
            opponent = self.opponents[opponent_idx]
            return opponent_idx, opponent
        
    def register_outcome(self, opponent_idx, outcome):
        self.history.append(opponent_idx)
        self.history_outcomes.append(outcome)
        self.t += 1




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =============================================================================
# load_sac_agent helper (given in your prompt)
# =============================================================================
def load_sac_agent(config_path, checkpoint_path, env):
    """
    Loads a SACAgent from a checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Try to get the config from the checkpoint; if not found, load from the config file.
    if "config" in checkpoint:
        config = checkpoint["config"]
        print("Loaded configuration from checkpoint.")
    else:
        with open(config_path, "r") as f:
            config = json.load(f)
        print("Loaded configuration from config file.")

    # If checkpoint is a tuple, convert it into a dictionary.
    if isinstance(checkpoint, tuple):
        tpl = checkpoint
        checkpoint = {}
        checkpoint["actor_state_dict"] = tpl[0]
        checkpoint["critic1_state_dict"] = tpl[1]
        checkpoint["critic2_state_dict"] = tpl[2]

    # Convert hidden sizes to lists of int if necessary
    if isinstance(config["hidden_sizes_actor"], str):
        config["hidden_sizes_actor"] = list(map(int, config["hidden_sizes_actor"].split(",")))
    if isinstance(config["hidden_sizes_critic"], str):
        config["hidden_sizes_critic"] = list(map(int, config["hidden_sizes_critic"].split(",")))

    # Convert learn_alpha to bool if it is a string
    learn_alpha = config["learn_alpha"]
    if isinstance(learn_alpha, str):
        learn_alpha = learn_alpha.lower() == "true"

    agent = SACAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        discount=config.get("discount", 0.99),
        buffer_size=config.get("buffer_size", int(1e6)),
        learning_rate_actor=config.get("learning_rate_actor", 1e-3),
        learning_rate_critic=config.get("learning_rate_critic", 1e-3),
        update_every=config.get("update_every", 1),
        use_per=config.get("use_per", False),
        use_ere=config.get("use_ere", False),
        per_alpha=config.get("per_alpha", 0.6),
        per_beta=config.get("per_beta", 0.4),
        ere_eta0=config.get("ere_eta0", 0.996),
        ere_c_k_min=config.get("ere_c_k_min", 2500),
        noise=config.get(
            "noise",
            {
                "type": "colored",
                "sigma": 0.1,
                "theta": 0.15,
                "dt": 1e-2,
                "beta": 1.0,
                "seq_len": 1000,
            },
        ),
        batch_size=config.get("batch_size", 256),
        hidden_sizes_actor=config["hidden_sizes_actor"],
        hidden_sizes_critic=config["hidden_sizes_critic"],
        tau=config.get("tau", 0.005),
        learn_alpha=learn_alpha,
        alpha=config.get("alpha", 0.2),
        control_half=True,  # True by default for hockey environment
    )

    # Restore the agent's full state
    agent.restore_full_state(checkpoint)
    return agent


def load_td3_agent(config_path, checkpoint_prefix, env):
    """
    Loads a TD3 agent from a JSON config and checkpoint prefix.

    The TD3 checkpoint is assumed to be saved as multiple files with the prefix, e.g.:
       - path/to/checkpoint_actor.pth
       - path/to/checkpoint_critic.pth
       - ...
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    print("Loaded TD3 configuration from:", config_path)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2
    max_action = env.action_space.high[0]

    # Create your TD3 agent
    agent = TD3(state_dim, action_dim, max_action, training_config=config)
    
    import copy
    agent.critic.load_state_dict(
        torch.load(checkpoint_prefix + "_critic.pth", map_location=device)
    )
    agent.critic_optimizer.load_state_dict(
        torch.load(checkpoint_prefix + "_critic_optimizer.pth", map_location=device)
    )
    agent.critic_target = copy.deepcopy(agent.critic)

    agent.actor.load_state_dict(
        torch.load(checkpoint_prefix + "_actor.pth", map_location=device)
    )
    agent.actor_optimizer.load_state_dict(
        torch.load(checkpoint_prefix + "_actor_optimizer.pth", map_location=device)
    )
    agent.actor_target = copy.deepcopy(agent.actor)

    if getattr(agent, 'use_rnd', False):
        agent.rnd.target_network.load_state_dict(
            torch.load(checkpoint_prefix + "_rnd_target.pth", map_location=device)
        )
        agent.rnd.predictor_network.load_state_dict(
            torch.load(checkpoint_prefix + "_rnd_predictor.pth", map_location=device)
        )
        agent.rnd.optimizer.load_state_dict(
            torch.load(checkpoint_prefix + "_rnd_optimizer.pth", map_location=device)
        )
    return agent


# =============================================================================
# Custom Classes
# =============================================================================

class CustomHockeyEnv:
    """
    Wraps the HockeyEnv to conform to the standard Gym API.
    """
    def __init__(self, mode=Mode.NORMAL, render_mode=None, reward="basic"):
        self.env = HockeyEnv(mode=mode, reward=reward)
        self.render_mode = render_mode

    def reset(self):
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        done = done or trunc  # combine done and truncation flags
        return obs, reward, done, info

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.set_seed(seed)
        self.env.action_space.seed(seed)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def get_info_agent_two(self):
        return self.env.get_info_agent_two()


# =============================================================================
# TrainedOpponent class
# =============================================================================
class TrainedOpponent:
    """
    An opponent that uses a trained agent (SAC or TD3) to select actions.
    """
    def __init__(
        self,
        agent=None,
        training=False,
        config_path=None,
        checkpoint_path=None,
        env=None,
        opponent_type="sac",   # "sac" or "td3"
    ):
        """
        If `agent` is not None, we directly use it.
        Otherwise, we load the agent from the appropriate loader:
          - if opponent_type == "sac", call load_sac_agent(...)
          - if opponent_type == "td3", call load_td3_agent(...)
        """
        self.training = training
        self.opponent_type = opponent_type.lower()

        if agent is not None:
            self.agent = agent
        else:
            if config_path is None or checkpoint_path is None or env is None:
                raise ValueError("Need config_path, checkpoint_path, and env to load an agent.")
            
            if self.opponent_type == "sac":
                loaded_agent = load_sac_agent(config_path, checkpoint_path, env)
            elif self.opponent_type == "td3":
                # For TD3, the checkpoint path is typically a prefix
                #   e.g. "my_dir/checkpoint" with multiple files
                loaded_agent = load_td3_agent(config_path, checkpoint_path, env)
            else:
                raise ValueError(f"Unknown opponent_type={self.opponent_type}, use 'sac' or 'td3'")
            
            self.agent = loaded_agent

    def act(self, observation, eval_mode=True, rollout=False):
        """
        For a SAC or TD3 opponent:
          - If SAC: use the previous logic (actor, sample, etc.)
          - If TD3: typically agent.act(obs, add_noise=False) for eval
        """
        obs_t = torch.FloatTensor(observation).unsqueeze(0).to(device)

        if self.opponent_type == "sac":
            # The existing SAC logic
            self.agent.actor.eval()
            with torch.no_grad():
                if eval_mode:
                    action_mean, _ = self.agent.actor(obs_t)
                    action = torch.tanh(action_mean) * self.agent.actor.action_scale + self.agent.actor.action_bias
                elif rollout:
                    action, _ = self.agent.actor.sample(obs_t, use_exploration_noise=True)
                else:
                    action, _ = self.agent.actor.sample(obs_t, use_exploration_noise=False)

            self.agent.actor.train(self.training)
            return action.cpu().numpy()[0]

        elif self.opponent_type == "td3":
            # For TD3, we typically do something like:
            #   agent.act(obs, add_noise=False) if eval_mode
            #   agent.act(obs, add_noise=True)  if rollout
            # The exact interface depends on your TD3 code.

            obs_np = observation  # shape [obs_dim,], no batch
            if eval_mode:
                # no noise
                action = self.agent.act(obs_np, add_noise=False)
            elif rollout:
                # some code for exploration
                action = self.agent.act(obs_np, add_noise=True)
            else:
                # e.g. training step, do no noise or something
                action = self.agent.act(obs_np, add_noise=False)

            return action  # or action.tolist() if you prefer

        else:
            raise ValueError("TrainedOpponent: unknown opponent_type in act() method")

def get_opponent(opponent_type, env, trained_agent=None):
    """
    Returns an opponent instance based on the specified type.
    For 'weak' or 'strong', returns a BasicOpponent; for 'trained', a TrainedOpponent.
    """
    if opponent_type == "weak":
        return BasicOpponent(weak=True)
    elif opponent_type == "strong":
        return BasicOpponent(weak=False)
    elif opponent_type == "trained":
        if trained_agent is None:
            raise ValueError("trained_agent must be provided for 'trained' opponent type.")
        return TrainedOpponent(trained_agent, training=False)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")


# =============================================================================
# SAC Trainer Class
# =============================================================================

class Trainer:
    def __init__(self, args):
        self.args = args
        self.setup_environment()
        self.setup_agent()
        self.setup_opponent()  # Create an opponent if requested.
        self.setup_logging()

    def setup_environment(self):
        mode = Mode.NORMAL
        if hasattr(self.args, "hockey_mode"):
            if isinstance(self.args.hockey_mode, str):
                mode = Mode[self.args.hockey_mode.upper()]
            else:
                mode = Mode(self.args.hockey_mode)

        keep_mode = self.args.keep_mode

        self.env = HockeyEnv(mode=mode, keep_mode=keep_mode, reward=self.args.reward)
        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            self.env.action_space.seed(self.args.seed)

    def load_all_sac_opponents_from_folder(self, folder_path, env, device=None):
        """
        Searches 'folder_path' for subfolders, each containing a 'config.json'
        and at least one .pth checkpoint. Loads each of these as a TrainedOpponent.
        
        Returns a list of TrainedOpponent objects.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folder = Path(folder_path)
        opponents = []

        if not folder.exists() or not folder.is_dir():
            print(f"[Warning] Opponents folder '{folder_path}' not found or not a directory.")
            return opponents

        # If you save multiple checkpoints, decide which one to load. 
        # For example, let's look for "checkpoint_episode_*.pth" or "final" etc.
        ckpt_candidates = list(folder.glob("*.pth"))
        if len(ckpt_candidates) == 0:
            return opponents

        # Let us pick the highest-episode checkpoint, or pick the first one, etc.
        ckpt_candidates.sort()
        checkpoint_file = ckpt_candidates[-1]  # e.g. last (highest) by name

        for checkpoint_file in ckpt_candidates:
            try:
                loaded_agent = load_sac_agent(
                    config_path="config.json",
                    checkpoint_path=str(checkpoint_file),
                    env=env
                )
                opp = TrainedOpponent(agent=loaded_agent, training=False)
                opponents.append(opp)
                print(f"Loaded opponent from {folder_path} -> {checkpoint_file.name}")
            except Exception as e:
                print(f"[Warning] Could not load from {folder_path}: {e}")

        return opponents

    def setup_agent(self):
        if self.args.self_play and self.args.sp_load:
            # We assume the user also gave us self.args.sp_agent_config
            # so we can reconstruct the agent with exactly the same hyperparams.
            print(f"Loading pre-trained agent from {self.args.sp_agent_checkpoint}")
            self.agent = load_sac_agent(
                config_path=self.args.sp_agent_config,
                checkpoint_path=self.args.sp_agent_checkpoint,
                env=self.env
            )
        else:
            hidden_actor = list(map(int, self.args.hidden_sizes_actor.split(",")))
            hidden_critic = list(map(int, self.args.hidden_sizes_critic.split(",")))
            learn_alpha_bool = self.args.learn_alpha.lower() == "true"

            self.agent = SACAgent(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                discount=self.args.discount,
                buffer_size=self.args.buffer_size,
                learning_rate_actor=self.args.lr,
                learning_rate_critic=self.args.lr,
                update_every=self.args.update_every,
                use_per=self.args.use_per,
                use_ere=self.args.use_ere,
                per_alpha=self.args.per_alpha,
                per_beta=self.args.per_beta,
                ere_eta0=self.args.ere_eta0,
                ere_c_k_min=self.args.ere_min_size,
                noise={
                    "type": self.args.noise_type,
                    "sigma": self.args.noise_sigma,
                    "theta": self.args.noise_theta,
                    "dt": self.args.noise_dt,
                    "beta": self.args.noise_beta,
                    "seq_len": self.args.noise_seq_len,
                },
                batch_size=self.args.batch_size,
                hidden_sizes_actor=hidden_actor,
                hidden_sizes_critic=hidden_critic,
                tau=self.args.tau,
                learn_alpha=learn_alpha_bool,
                alpha=self.args.alpha,
                control_half=True   # True by default for hockey environment
            )

    def setup_opponent(self):
        """
        Create an opponent if the command-line argument for opponent_type is not "none".
        """
        if self.args.opponent_type.lower() == "none":
            self.opponent = None
        elif self.args.opponent_type.lower() in ["weak", "weak_basic"]:
            self.opponent = BasicOpponent(weak=True, keep_mode=self.args.keep_mode)
        elif self.args.opponent_type.lower() == "strong":
            self.opponent = BasicOpponent(weak=False, keep_mode=self.args.keep_mode)
        elif self.args.opponent_type.lower() == "trained":
            # If we want a pre-trained opponent, it must be loaded from a checkpoint/config
            # (If your usage is different, adapt here)
            if not self.args.sp_opponent_checkpoint or not self.args.sp_opponent_config:
                raise ValueError("Must provide --sp_opponent_checkpoint and --sp_opponent_config for a trained opponent.")
            # Create a test environment for loading the agent
            test_env = HockeyEnv(mode=self.env.mode, keep_mode=self.args.keep_mode, reward=self.args.reward)
            self.opponent = TrainedOpponent(
                agent=None,
                training=False,
                config_path=self.args.sp_opponent_config,
                checkpoint_path=self.args.sp_opponent_checkpoint,
                env=test_env
            )
        else:
            self.opponent = None

        # Additional specialized opponents if in certain hockey modes:
        if self.args.hockey_mode.lower() == "train_defense":
            self.opponent = BasicAttackOpponent(keep_mode=self.args.keep_mode)
        elif self.args.hockey_mode.lower() == "train_shooting":
            self.opponent = BasicDefenseOpponent(keep_mode=self.args.keep_mode)

    def get_run_name(self):
        if self.args.name != "SAC":
            return str(self.args.id) + "_" + self.args.name
        components = [
            f"{self.args.id}",
            f"{self.args.env_name}",
            f"lr:{self.args.lr}",
            f"seed:{self.args.seed}" if self.args.seed is not None else "seed:none",
        ]
        if self.args.use_per:
            components.append(f"PER-a:{self.args.per_alpha}-b:{self.args.per_beta}")
        if self.args.use_ere:
            components.append(f"ERE-eta:{self.args.ere_eta0}")
        timestamp = datetime.now().strftime("%d.%m.%Y._%H:%M")
        components.append(timestamp)
        return "_".join(components)

    def setup_logging(self):
        self.run_name = self.get_run_name()
        self.output_dir = Path(self.args.output_dir) / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "training-log.txt"
        with open(self.log_file, "w") as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write(f"PID: {self.args.id}\n")
            f.write("-" * 50 + "\n")
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(vars(self.args), f, indent=4)
        # Initialize metrics.
        self.rewards = []
        self.lengths = []
        self.losses = []
        self.timestep = 0
        self.eval_win_ratios = []
        self.eval_episodes_done = []
        self.create_summary_file()
        np.savez(
            self.output_dir / "rewards.npz",
            rewards=np.array([]),
            episode_lengths=np.array([]),
            moving_avg_100=np.array([]),
            config=json.dumps(vars(self.args)),
        )

    def create_summary_file(self):
        # Opponent description for logging
        opponent_desc = "None"
        if self.opponent is not None:
            if self.args.opponent_type.lower() in ["weak", "weak_basic"]:
                opponent_desc = "BasicOpponent (Weak)"
            elif self.args.opponent_type.lower() == "strong":
                opponent_desc = "BasicOpponent (Strong)"
            elif self.args.opponent_type.lower() == "trained":
                opponent_desc = "TrainedOpponent"
            else:
                opponent_desc = "Unknown"

        summary = [
            "Experiment Summary",
            "=" * 50,
            "\nEnvironment Configuration:",
            f"  Run Name: {self.run_name}",
            f"  Environment: {self.args.env_name}",
            f"  Max Episodes: {self.args.max_episodes}",
            f"  Max Timesteps per Episode: {self.args.max_timesteps}",
            f"  Reward Type: {self.args.reward}",
            "\nTraining Configuration:",
            f"  Learning Rate: {self.args.lr}",
            f"  Update Frequency: {self.args.update_every}",
            f"  Random Seed: {self.args.seed}",
            f"  Discount Factor: {self.args.discount}",
            f"  Replay Buffer Size: {self.args.buffer_size}",
            f"  Batch Size: {self.args.batch_size}",
            f"  Actor Hidden Layers: {self.args.hidden_sizes_actor}",
            f"  Critic Hidden Layers: {self.args.hidden_sizes_critic}",
            f"  Tau: {self.args.tau}",
            f"  Alpha: {self.args.alpha}",
            f"  Learn Alpha?: {self.args.learn_alpha}",
            f"  Keep Mode?: {self.args.keep_mode}",
            "\nPER Configuration:",
            f"  Enabled: {self.args.use_per}",
            f"  PER Alpha: {self.args.per_alpha}",
            f"  PER Beta: {self.args.per_beta}",
            "\nERE Configuration:",
            f"  Enabled: {self.args.use_ere}",
            f"  ERE Eta0: {self.args.ere_eta0}",
            f"  ERE Min Size: {self.args.ere_min_size}",
            "\nNoise Configuration:",
            f"  Noise Type: {self.args.noise_type}",
            f"  Noise Sigma: {self.args.noise_sigma}",
            f"  Noise Theta: {self.args.noise_theta}" if self.args.noise_type == "ornstein" else "",
            f"  Noise dt: {self.args.noise_dt}" if self.args.noise_type == "ornstein" else "",
            f"  Noise Beta: {self.args.noise_beta}" if self.args.noise_type == "colored" else "",
            f"  Noise Seq Len: {self.args.noise_seq_len}" if self.args.noise_type in ["colored", "pink"] else "",
            "\nLogging Configuration:",
            f"  Output Directory: {self.args.output_dir}",
            f"  Save Interval: {self.args.save_interval}",
            f"  Log Interval: {self.args.log_interval}",
            f"  Eval Interval: {self.args.eval_interval}",
            f"  Eval Episodes: {self.args.eval_episodes}",
            "\nOpponent Configuration:",
            f"  Opponent Type: {self.args.opponent_type}",
            f"  Opponent Details: {opponent_desc}",
            "\nSelf-Play:",
            f"  Self-Play Enabled?: {self.args.self_play}",
            f"  SP Min Epochs: {getattr(self.args, 'sp_min_epochs', None)}",
            f"  SP Threshold: {getattr(self.args, 'sp_threshold', None)}",
            "\nRun Name:",
            f"  {self.run_name}",
        ]
        summary = [s for s in summary if s.strip()]
        with open(self.output_dir / "experiment_summary.txt", "w") as f:
            f.write("\n".join(summary))

    def save_checkpoint(self, episode):
        checkpoint_path = self.output_dir / f"checkpoint_episode_{episode}.pth"
        torch.save(self.agent.full_state(), checkpoint_path)
        minimal_ckpt = {
            "actor_state_dict": self.agent.actor.state_dict(),
            "critic1_state_dict": self.agent.critic1.state_dict(),
            "critic2_state_dict": self.agent.critic2.state_dict()
        }
        # Optionally: torch.save(minimal_ckpt, self.output_dir / f"weights_only_episode_{episode}.pth")

        stats = {
            "rewards": self.rewards,
            "lengths": self.lengths,
            "losses": self.losses,
            "config": vars(self.args),
            "buffer_stats": (
                self.agent.buffer.get_statistics()
                if hasattr(self.agent.buffer, "get_statistics")
                else None
            ),
            "eval_win_ratios": self.eval_win_ratios,
            "eval_episodes_done": self.eval_episodes_done,
        }
        with open(self.output_dir / f"statistics_episode_{episode}.pkl", "wb") as f:
            pickle.dump(stats, f)
        moving_avg = (
            np.convolve(self.rewards, np.ones(100) / 100, mode="valid")
            if len(self.rewards) >= 100
            else np.array([])
        )
        np.savez(
            self.output_dir / "rewards.npz",
            rewards=np.array(self.rewards),
            episode_lengths=np.array(self.lengths),
            moving_avg_100=moving_avg,
            config=json.dumps(vars(self.args)),
            eval_win_ratios=self.eval_win_ratios,
            eval_episodes_done=self.eval_episodes_done,
        )

    def plot_metrics(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards, label="Episode Reward")
        plt.title(f"Episode Rewards - {self.args.env_name}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.savefig(self.output_dir / "rewards.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.lengths, label="Episode Length")
        plt.title(f"Episode Lengths - {self.args.env_name}")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.legend()
        plt.savefig(self.output_dir / "lengths.png")
        plt.close()

        if self.losses:
            recent_losses = np.array(self.losses[-1000:])
            plt.figure(figsize=(10, 5))
            plt.plot(recent_losses[:, 0], label="Critic1")
            plt.plot(recent_losses[:, 1], label="Critic2")
            plt.plot(recent_losses[:, 2], label="Actor")
            plt.title(f"Recent Training Losses - {self.args.env_name}")
            plt.xlabel("Step (last 1000 updates)")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(self.output_dir / "losses.png")
            plt.close()

        plt.figure(figsize=(10, 5))
        window_size = 100
        if len(self.rewards) >= window_size:
            mov_avg = np.convolve(self.rewards, np.ones(window_size) / window_size, mode="valid")
            plt.plot(np.arange(len(mov_avg)) + window_size, mov_avg, label="MA(100)")
        else:
            plt.plot(self.rewards, label="Episode Reward")
        plt.title(f"Moving Average Reward (window={window_size}) - {self.args.env_name}")
        plt.xlabel("Episode")
        plt.ylabel("Avg Reward")
        plt.legend()
        plt.savefig(self.output_dir / "rewards_moving_avg.png")
        plt.close()

        if len(self.eval_win_ratios) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(self.eval_episodes_done, self.eval_win_ratios, marker="o", label="Win Ratio")
            plt.title("Evaluation Win Ratio")
            plt.xlabel("Training Episode")
            plt.ylabel("Win Ratio (0 to 1)")
            plt.legend()
            plt.savefig(self.output_dir / "win_ratio.png")
            plt.close()
            
        if hasattr(self, "eval_episodes_done_3") and len(self.eval_episodes_done_3) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(self.eval_episodes_done_3, self.eval_opponent_ratios, 'r-o', label="vs Fixed Opp.")
            plt.plot(self.eval_episodes_done_3, self.eval_weak_ratios, 'g-x', label="vs Weak Opp.")
            plt.plot(self.eval_episodes_done_3, self.eval_strong_ratios, 'b--s', label="vs Strong Opp.")
            plt.title("Win Ratios in Mode3 Self-Play")
            plt.xlabel("Training Episode")
            plt.ylabel("Win Ratio")
            plt.legend()
            plt.savefig(self.output_dir / "mode3_winratios.png")
            plt.close()
    # ----------------------------------------------------
    # Default training loop (if not using self-play)
    # ----------------------------------------------------
    def train(self):
        for episode in range(1, self.args.max_episodes + 1):
            self.agent.K = 0
            self.agent.reset_noise()
            obs, _info = self.env.reset()
            total_reward = 0.0

            for t in range(self.args.max_timesteps):
                self.timestep += 1
                agent_action = self.agent.act(obs, eval_mode=False, rollout=True)
                if self.opponent is not None:
                    # get opponent action
                    opponent_obs = self.env.obs_agent_two() if hasattr(self.env, "obs_agent_two") else obs
                    if isinstance(self.opponent, TrainedOpponent):
                        opponent_action = self.opponent.act(opponent_obs)
                    else:
                        opponent_action = self.opponent.act(opponent_obs)
                    full_action = np.hstack([agent_action, opponent_action])
                else:
                    opponent_action = np.array([0, 0, 0, 0], dtype=np.float32)
                    full_action = np.hstack([agent_action, opponent_action])

                next_obs, reward, done, trunc, info = self.env.step(full_action)
                total_reward += reward
                self.agent.store_transition((obs, agent_action, reward, next_obs, done))

                if(self.args.mirror):
                    mirrored_obs = self.env.mirror_state(obs)
                    mirrored_agent_action = self.env.mirror_action(agent_action)
                    mirrored_next_obs = self.env.mirror_state(next_obs)
                    self.agent.store_transition((mirrored_obs, mirrored_agent_action, reward, mirrored_next_obs, done))
                
                self.agent.K += 1
                obs = next_obs
                if done or trunc:
                    break

            self.losses.extend(self.agent.train(self.agent.K))
            self.rewards.append(total_reward)
            self.lengths.append(t)

            # Evaluation
            if self.args.eval_interval > 0 and episode % self.args.eval_interval == 0:
                ratio = self.evaluate_policy(eval_episodes=self.args.eval_episodes)
                self.eval_win_ratios.append(ratio)
                self.eval_episodes_done.append(episode)
                eval_msg = f"[Eval] Episode {episode} -> Win Ratio: {ratio:.3f}"
                print(eval_msg)
                with open(self.log_file, "a") as f:
                    f.write(eval_msg + "\n")

            # Checkpoint & Logging
            if episode % self.args.save_interval == 0:
                self.save_checkpoint(episode)
                self.plot_metrics()

            if episode % self.args.log_interval == 0:
                avg_reward = np.mean(self.rewards[-self.args.log_interval:])
                avg_length = int(np.mean(self.lengths[-self.args.log_interval:]))
                log_msg = f"Episode {episode} \t avg length: {avg_length} \t reward: {avg_reward:.2f}"
                print(log_msg)
                with open(self.log_file, "a") as f:
                    f.write(log_msg + "\n")

        self.save_checkpoint("final")
        self.plot_metrics()

    
    def copy_agent(self, agent_or_opponent):
        """
        Make a 'clone' of the given object:
          - If it's a BasicOpponent, return it as-is (no copying).
          - If it's a TrainedOpponent, clone the underlying SACAgent.
          - If it's already an SACAgent, clone directly.
        Returns a TrainedOpponent if it's an agent, or BasicOpponent as-is.
        """
        # <-- ADDED
        # 1) If it's BasicOpponent, just reuse it (no internal state to copy).
        if isinstance(agent_or_opponent, BasicOpponent):
            return agent_or_opponent  # returning the same instance

        # 2) If it's a TrainedOpponent, get the underlying SACAgent.
        if isinstance(agent_or_opponent, TrainedOpponent):
            agent_or_opponent = agent_or_opponent.agent

        # 3) Now agent_or_opponent is presumably a SACAgent
        temp_ckpt = agent_or_opponent.full_state()

        # Re-derive hyperparameters
        hidden_actor = list(map(int, self.args.hidden_sizes_actor.split(",")))
        hidden_critic = list(map(int, self.args.hidden_sizes_critic.split(",")))
        learn_alpha_bool = (self.args.learn_alpha.lower() == "true")

        new_opponent_snapshot = SACAgent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            discount=self.args.discount,
            buffer_size=self.args.buffer_size,
            learning_rate_actor=self.args.lr,
            learning_rate_critic=self.args.lr,
            update_every=self.args.update_every,
            use_per=self.args.use_per,
            use_ere=self.args.use_ere,
            per_alpha=self.args.per_alpha,
            per_beta=self.args.per_beta,
            ere_eta0=self.args.ere_eta0,
            ere_c_k_min=self.args.ere_min_size,
            noise={
                "type": self.args.noise_type,
                "sigma": self.args.noise_sigma,
                "theta": self.args.noise_theta,
                "dt": self.args.noise_dt,
                "beta": self.args.noise_beta,
                "seq_len": self.args.noise_seq_len,
            },
            batch_size=self.args.batch_size,
            hidden_sizes_actor=hidden_actor,
            hidden_sizes_critic=hidden_critic,
            tau=self.args.tau,
            learn_alpha=learn_alpha_bool,
            alpha=self.args.alpha,
            control_half=True
        )
        new_opponent_snapshot.restore_full_state(temp_ckpt)
        return TrainedOpponent(agent=new_opponent_snapshot, training=False)
    # ----------------------------------------------------
    # SELF-PLAY TRAINING LOOP
    # ----------------------------------------------------
    def train_self_play(self):
        """
        Self-play mode 1:
        1) Train for sp_min_epochs with the current (fixed) opponent
        2) If average reward >= sp_threshold, update the opponent
        3) Possibly switch with old versions or the 'strong' built-in
        4) Mirror states if self.args.mirror is True
        """
        old_opponents = []
        best_avg_reward = -9999.0
        episodes_since_update = 0
        episode = 0

        while episode < self.args.max_episodes:
            sp_min = self.args.sp_min_epochs
            local_rewards = []

            for local_ep in range(sp_min):
                episode += 1
                obs, _info = self.env.reset()
                self.agent.K = 0
                total_reward = 0.0

                for t in range(self.args.max_timesteps):
                    agent_action = self.agent.act(obs, eval_mode=False, rollout=True)

                    if self.opponent is not None:
                        opponent_obs = self.env.obs_agent_two() if hasattr(self.env, "obs_agent_two") else obs
                        opponent_action = self.opponent.act(opponent_obs)
                        full_action = np.hstack([agent_action, opponent_action])
                    else:
                        full_action = np.hstack([agent_action, np.zeros(4, dtype=np.float32)])

                    next_obs, reward, done, trunc, info = self.env.step(full_action)
                    total_reward += reward
                    self.agent.store_transition((obs, agent_action, reward, next_obs, done))

                    # ---- Mirror augmentation if enabled ----
                    if self.args.mirror:
                        mirrored_obs = self.env.mirror_state(obs)
                        mirrored_agent_action = self.env.mirror_action(agent_action)
                        mirrored_next_obs = self.env.mirror_state(next_obs)
                        self.agent.store_transition((mirrored_obs, mirrored_agent_action, reward, mirrored_next_obs, done))
                    # ----------------------------------------

                    self.agent.K += 1
                    obs = next_obs
                    if done or trunc:
                        break

                # After each episode, do training updates
                self.losses.extend(self.agent.train(self.agent.K))
                self.rewards.append(total_reward)
                self.lengths.append(t)
                local_rewards.append(total_reward)

                # Evaluate if needed
                if self.args.eval_interval > 0 and episode % self.args.eval_interval == 0:
                    ratio = self.evaluate_policy(eval_episodes=self.args.eval_episodes)
                    self.eval_win_ratios.append(ratio)
                    self.eval_episodes_done.append(episode)
                    eval_msg = f"[Eval] Episode {episode} -> Win Ratio: {ratio:.3f}"
                    print(eval_msg)
                    with open(self.log_file, "a") as f:
                        f.write(eval_msg + "\n")

                # Checkpoint & plots
                if episode % self.args.save_interval == 0:
                    self.save_checkpoint(episode)
                    self.plot_metrics()

                # Logging
                if episode % self.args.log_interval == 0:
                    avg_reward = np.mean(self.rewards[-self.args.log_interval:])
                    avg_length = int(np.mean(self.lengths[-self.args.log_interval:]))
                    log_msg = f"Episode {episode} \t avg length: {avg_length} \t reward: {avg_reward:.2f}"
                    print(log_msg)
                    with open(self.log_file, "a") as f:
                        f.write(log_msg + "\n")

                if episode >= self.args.max_episodes:
                    break

            # After sp_min_epochs, check average reward
            mean_local = np.mean(local_rewards)
            if mean_local >= self.args.sp_threshold:
                print(f"Updating opponent!  Average Reward over last {sp_min} = {mean_local:.2f}")
                old_opponents.append(self.copy_agent(self.opponent))
                new_opponent_snapshot = self.copy_agent(self.agent)
                self.opponent = new_opponent_snapshot
                best_avg_reward = mean_local
                episodes_since_update = 0
            else:
                episodes_since_update += sp_min

            # If no improvement for 5 cycles, break
            if episodes_since_update >= (5 * sp_min):
                print("No improvement for 5 cycles; stopping self-play.")
                break

            # With low prob, switch the opponent with old version or basic strong
            if random.random() < self.args.sp_switch_prob and (old_opponents or self.args.opponent_type.lower() != "none"):
                candidates = []
                candidates.append(BasicOpponent(weak=False, keep_mode=self.args.keep_mode))
                candidates += old_opponents
                chosen = random.choice(candidates)
                self.opponent = chosen

        self.save_checkpoint("final")
        self.plot_metrics()


    def train_self_play_mode2(self):
        """
        Self-play mode 2 with PrioritizedOpponentBuffer (D-UCB).
        We pick an opponent each episode from the buffer, do mirror if self.args.mirror,
        and if the agent's win rate vs all opponents >= threshold, add a new clone.
        """
        self.ducb_buffer = PrioritizedOpponentBuffer(B=1, xi=1, gamma=0.95, tau=1000)

        # Add initial opponents
        opp_weak = BasicOpponent(weak=True, keep_mode=self.args.keep_mode)
        opp_strong = BasicOpponent(weak=False, keep_mode=self.args.keep_mode)
        self.ducb_buffer.add_opponent(opp_weak)
        self.ducb_buffer.add_opponent(opp_strong)

        wr_opponent_thresh = self.args.sp_wr_threshold
        n_update = self.args.sp_n_update
        episodes_since_sampling = 0

        episode = 0
        while episode < self.args.max_episodes:
            for local_ep in range(self.args.sp_min_epochs):
                episode += 1
                obs, _info = self.env.reset()
                self.agent.K = 0
                total_reward = 0.0

                opp_idx, opponent = self.ducb_buffer.get_opponent()

                # Run one episode
                for t in range(self.args.max_timesteps):
                    agent_action = self.agent.act(obs, eval_mode=False, rollout=True)

                    opponent_obs = self.env.obs_agent_two() if hasattr(self.env, "obs_agent_two") else obs
                    opp_action = opponent.act(opponent_obs)

                    full_action = np.hstack([agent_action, opp_action])
                    next_obs, reward, done, trunc, info = self.env.step(full_action)
                    total_reward += reward
                    self.agent.store_transition((obs, agent_action, reward, next_obs, done))

                    # Mirror transitions if requested
                    if self.args.mirror:
                        mirrored_obs = self.env.mirror_state(obs)
                        mirrored_agent_action = self.env.mirror_action(agent_action)
                        mirrored_next_obs = self.env.mirror_state(next_obs)
                        self.agent.store_transition((mirrored_obs, mirrored_agent_action, reward, mirrored_next_obs, done))

                    self.agent.K += 1
                    obs = next_obs
                    if done or trunc:
                        break

                # Train agent
                self.losses.extend(self.agent.train(self.agent.K))
                self.rewards.append(total_reward)
                self.lengths.append(t)

                # Register outcome in D-UCB
                outcome = 0.0
                if "winner" in info:
                    if info["winner"] == 1:
                        outcome = 1.0
                    elif info["winner"] == 0:
                        outcome = 0.5
                    else:
                        outcome = 0.0
                self.ducb_buffer.register_outcome(opp_idx, outcome)

                # Possibly re-sample after n_update
                episodes_since_sampling += 1
                if episodes_since_sampling >= n_update:
                    episodes_since_sampling = 0
                    # Next iteration's get_opponent() will pick another

                # Evaluate vs all existing opponents if needed
                if self.args.eval_interval > 0 and episode % self.args.eval_interval == 0:
                    wr_weak = self.evaluate_win_rate(self.agent, opp_weak, self.args.eval_episodes)
                    update_opponent = (wr_weak >= wr_opponent_thresh)

                    if self.ducb_buffer.K > 1 and update_opponent:
                        for idx in range(1, self.ducb_buffer.K):
                            op = self.ducb_buffer.opponents[idx]
                            wr_i = self.evaluate_win_rate(self.agent, op, self.args.eval_episodes)
                            if wr_i < wr_opponent_thresh:
                                update_opponent = False
                                break

                    if update_opponent:
                        print(f"[SP2] Episode {episode}: Win rate above threshold {wr_opponent_thresh}, adding new clone.")
                        new_clone = self.copy_agent(self.agent)
                        self.ducb_buffer.add_opponent(new_clone)

                    msg = f"[Eval] Episode {episode} -> wr_weak: {wr_weak:.3f}"
                    print(msg)
                    with open(self.log_file, "a") as f:
                        f.write(msg + "\n")

                if episode % self.args.save_interval == 0:
                    self.save_checkpoint(episode)
                    self.plot_metrics()

                if episode % self.args.log_interval == 0:
                    avg_reward = np.mean(self.rewards[-self.args.log_interval:])
                    avg_length = int(np.mean(self.lengths[-self.args.log_interval:]))
                    log_msg = f"Episode {episode} (SP-mode2) avg_length={avg_length}, reward={avg_reward:.2f}"
                    print(log_msg)
                    with open(self.log_file, "a") as f:
                        f.write(log_msg + "\n")

                if episode >= self.args.max_episodes:
                    break

        self.save_checkpoint("final")
        self.plot_metrics()
        
    def train_self_play_mode3(self):
        """
        Third mode of self-play, using exactly two fixed agents:
        - Our 'trainable' agent (self.agent)
        - A separate 'opponent' loaded from sp_opponent_checkpoint + config
        Then in each evaluation phase, we measure win rates vs:
        1) the fixed opponent,
        2) a built-in weak opponent,
        3) a built-in strong opponent.
        We store all three lines in self.eval_opponent_ratios, self.eval_weak_ratios, self.eval_strong_ratios
        so we can see them in the final plot.
        """

        if not self.args.sp_opponent_checkpoint or not self.args.sp_opponent_config:
            raise ValueError("Mode 3: Must provide --sp_opponent_checkpoint and --sp_opponent_config.")

        print("[Mode 3] Loaded a second agent as the fixed opponent for training.\n")

        # We'll store three lines of evaluation in arrays, so we can plot them:
        self.eval_opponent_ratios = []
        self.eval_weak_ratios = []
        self.eval_strong_ratios = []
        self.eval_episodes_done_3 = []  # For x-axis in the triple-plot
        # evaluate before starting:
        ratio_fixed = self.evaluate_win_rate(self.agent, self.opponent, self.args.eval_episodes)
        ratio_weak  = self.evaluate_win_rate(self.agent, BasicOpponent(weak=True), self.args.eval_episodes)
        ratio_strong = self.evaluate_win_rate(self.agent, BasicOpponent(weak=False), self.args.eval_episodes)
        episode = 0

        self.eval_opponent_ratios.append(ratio_fixed)
        self.eval_weak_ratios.append(ratio_weak)
        self.eval_strong_ratios.append(ratio_strong)
        self.eval_episodes_done_3.append(episode)

        eval_msg = (f"[Eval] Episode {episode} -> vsFixed={ratio_fixed:.3f}, "
                    f"vsWeak={ratio_weak:.3f}, vsStrong={ratio_strong:.3f}")
        print(eval_msg)
        with open(self.log_file, "a") as f:
            f.write(eval_msg + "\n")
        # 2) Standard training loop
        for episode in range(1, self.args.max_episodes + 1):
            self.agent.K = 0
            self.agent.reset_noise()
            obs, _info = self.env.reset()
            total_reward = 0.0

            for t in range(self.args.max_timesteps):
                self.timestep += 1
                agent_action = self.agent.act(obs, eval_mode=False, rollout=True)

                # Opponent is the fixed second agent
                opponent_obs = self.env.obs_agent_two() if hasattr(self.env, "obs_agent_two") else obs
                opponent_action = self.opponent.act(opponent_obs)

                full_action = np.hstack([agent_action, opponent_action])
                next_obs, reward, done, trunc, info = self.env.step(full_action)
                total_reward += reward
                self.agent.store_transition((obs, agent_action, reward, next_obs, done))

                # Mirror if asked
                if(self.args.mirror):
                    mirrored_obs = self.env.mirror_state(obs)
                    mirrored_agent_action = self.env.mirror_action(agent_action)
                    mirrored_next_obs = self.env.mirror_state(next_obs)
                    self.agent.store_transition((mirrored_obs, mirrored_agent_action, reward, mirrored_next_obs, done))

                self.agent.K += 1
                obs = next_obs
                if done or trunc:
                    break

            # After the episode, do training updates
            self.losses.extend(self.agent.train(self.agent.K))
            self.rewards.append(total_reward)
            self.lengths.append(t)

            # 3) Evaluate triple if it's time
            if self.args.eval_interval > 0 and episode % self.args.eval_interval == 0:
                ratio_fixed = self.evaluate_win_rate(self.agent, self.opponent, self.args.eval_episodes)
                ratio_weak  = self.evaluate_win_rate(self.agent, BasicOpponent(weak=True), self.args.eval_episodes)
                ratio_strong = self.evaluate_win_rate(self.agent, BasicOpponent(weak=False), self.args.eval_episodes)

                self.eval_opponent_ratios.append(ratio_fixed)
                self.eval_weak_ratios.append(ratio_weak)
                self.eval_strong_ratios.append(ratio_strong)
                self.eval_episodes_done_3.append(episode)

                eval_msg = (f"[Eval] Episode {episode} -> vsFixed={ratio_fixed:.3f}, "
                            f"vsWeak={ratio_weak:.3f}, vsStrong={ratio_strong:.3f}")
                print(eval_msg)
                with open(self.log_file, "a") as f:
                    f.write(eval_msg + "\n")

            # 4) Checkpoint & Logging
            if episode % self.args.save_interval == 0:
                self.save_checkpoint(episode)
                self.plot_metrics()

            if episode % self.args.log_interval == 0:
                avg_reward = np.mean(self.rewards[-self.args.log_interval:])
                avg_length = int(np.mean(self.lengths[-self.args.log_interval:]))
                log_msg = f"Episode {episode} \t avg length: {avg_length} \t reward: {avg_reward:.2f}"
                print(log_msg)
                with open(self.log_file, "a") as f:
                    f.write(log_msg + "\n")

        # 5) after done
        self.save_checkpoint("final")
        self.plot_metrics()

    def train_self_play_mode4(self):
        """
        Self-play mode 4:
        1) Create a PrioritizedOpponentBuffer (D-UCB).
        2) Add basic weak + strong opponents.
        3) Load all SAC opponents found in --sp_opponents_folder, add them to the buffer.
        4) Train the main agent as in mode2. 
        If agent's win rate >= sp_wr_threshold vs *all* existing opponents,
        add a new clone of the agent to the buffer.
        """

        # 1) Create the D-UCB buffer
        self.ducb_buffer = PrioritizedOpponentBuffer(
            B=1, 
            xi=1, 
            gamma=0.95, 
            tau=1000
        )

        # 2) Add the basic opponents
        opp_weak = BasicOpponent(weak=True, keep_mode=self.args.keep_mode)
        opp_strong = BasicOpponent(weak=False, keep_mode=self.args.keep_mode)
        opp_basicattack = BasicAttackOpponent(keep_mode=self.args.keep_mode)
        opp_basicdefense = BasicDefenseOpponent(keep_mode=self.args.keep_mode)
        self.ducb_buffer.add_opponent(opp_weak)
        self.ducb_buffer.add_opponent(opp_strong)
        self.ducb_buffer.add_opponent(opp_basicattack)
        self.ducb_buffer.add_opponent(opp_basicdefense)
        

        # 3) Load any SAC opponents from folder
        if self.args.sp_opponents_folder.strip():
            loaded_opponents = self.load_all_sac_opponents_from_folder(
                self.args.sp_opponents_folder, 
                env=self.env
            )
            
  
        for opp in loaded_opponents:
            self.ducb_buffer.add_opponent(opp)

        # Gather parameters:
        wr_opponent_thresh = self.args.sp_wr_threshold   # e.g. 0.95
        n_update = self.args.sp_n_update                # e.g. 1000
        episode = 0
        episodes_since_sampling = 0
        
        # evaluate initial win rates:
        for idx_op, op in enumerate(self.ducb_buffer.opponents):
            wr_i = self.evaluate_win_rate(self.agent, op, self.args.eval_episodes)
            msg = f"[Eval] Start: vs -> WR={wr_i:.3f}"
            print(msg)
            with open(self.log_file, "a") as f:
                f.write(msg + "\n")

        # 4) Training Loop
        while episode < self.args.max_episodes:
            for local_ep in range(self.args.sp_min_epochs):
                episode += 1
                obs, _info = self.env.reset()
                self.agent.K = 0
                total_reward = 0.0

                # Get an opponent from the buffer using the D-UCB logic
                opp_idx, opponent = self.ducb_buffer.get_opponent()

                for t in range(self.args.max_timesteps):
                    agent_action = self.agent.act(obs, eval_mode=False, rollout=True)

                    if hasattr(self.env, "obs_agent_two"):
                        opp_obs = self.env.obs_agent_two()
                    else:
                        opp_obs = obs
                    opp_action = opponent.act(opp_obs)

                    full_action = np.hstack([agent_action, opp_action])
                    next_obs, reward, done, trunc, info = self.env.step(full_action)
                    total_reward += reward

                    self.agent.store_transition((obs, agent_action, reward, next_obs, done))

                    # Mirror augmentation if requested
                    if self.args.mirror:
                        mirrored_obs = self.env.mirror_state(obs)
                        mirrored_agent_action = self.env.mirror_action(agent_action)
                        mirrored_next_obs = self.env.mirror_state(next_obs)
                        self.agent.store_transition(
                            (mirrored_obs, mirrored_agent_action, reward, mirrored_next_obs, done)
                        )

                    self.agent.K += 1
                    obs = next_obs
                    if done or trunc:
                        break

                # After the episode, train the agent
                self.losses.extend(self.agent.train(self.agent.K))
                self.rewards.append(total_reward)
                self.lengths.append(t)

                # Register the outcome in D-UCB
                # If your environment sets info["winner"] = 1,0,-1 for agent, draw, or opponent:
                outcome = 0.0
                if "winner" in info:
                    if info["winner"] == 1:
                        outcome = 1.0
                    elif info["winner"] == 0:
                        outcome = 0.5
                    else:
                        outcome = 0.0
                self.ducb_buffer.register_outcome(opp_idx, outcome)

                # Possibly evaluate and see if we can add a new clone
                episodes_since_sampling += 1
                if episodes_since_sampling >= n_update:
                    episodes_since_sampling = 0
                    # Evaluate the agent vs all opponents in the buffer
                    all_good = True
                    for idx_op, op in enumerate(self.ducb_buffer.opponents):
                        wr_i = self.evaluate_win_rate(self.agent, op, self.args.eval_episodes)
                        if wr_i < wr_opponent_thresh:
                            all_good = False
                            break
                    if all_good:
                        print(f"[Mode4] Episode {episode}: Agent >= {wr_opponent_thresh} vs all. Adding new clone.")
                        new_clone = self.copy_agent(self.agent)
                        self.ducb_buffer.add_opponent(new_clone)

                # Also do a more general evaluation if desired
                if self.args.eval_interval > 0 and episode % self.args.eval_interval == 0:
                    # Example: Evaluate vs the strong baseline to track progress
                    wr_strong = self.evaluate_win_rate(self.agent, opp_strong, self.args.eval_episodes)
                    msg = f"[Eval] Episode {episode} vs Strong -> WR={wr_strong:.3f}"
                    print(msg)
                    with open(self.log_file, "a") as f:
                        f.write(msg + "\n")

                # Checkpoint & logging 
                if episode % self.args.save_interval == 0:
                    self.save_checkpoint(episode)
                    self.plot_metrics()

                if episode % self.args.log_interval == 0:
                    avg_reward = np.mean(self.rewards[-self.args.log_interval:])
                    avg_length = int(np.mean(self.lengths[-self.args.log_interval:]))
                    log_msg = (
                        f"[Mode4] Episode {episode} avg_length={avg_length}, reward={avg_reward:.2f}"
                    )
                    print(log_msg)
                    with open(self.log_file, "a") as f:
                        f.write(log_msg + "\n")

                if episode >= self.args.max_episodes:
                    break

        # Done
        self.save_checkpoint("final")
        self.plot_metrics()



    def evaluate_win_rate(self, agent, opponent, eval_episodes=100, render=False):
        """
        Evaluate the fraction of episodes that 'agent' wins against 'opponent'
        over 'eval_episodes' episodes. Essentially a 'win rate'.
        If your environment sets info["winner"] = +1 for agent, -1 for opponent, 0 if no winner/draw.
        """
        wins = 0
        for _ in range(eval_episodes):
            obs, _ = self.env.reset()
            done = False
            while not done:
                agent_action = agent.act(obs, eval_mode=True)
                opp_obs = self.env.obs_agent_two() if hasattr(self.env, "obs_agent_two") else obs
                opp_action = opponent.act(opp_obs)
                full_action = np.hstack([agent_action, opp_action])
                obs, reward, done, trunc, info = self.env.step(full_action)
                done = done or trunc
                if render:
                    self.env.render("human")

            if "winner" in info and info["winner"] == 1:
                wins += 1
        return float(wins) / eval_episodes



    def evaluate_policy(self, eval_episodes=1000):
        """
        Runs evaluation episodes in parallel and returns the win ratio.
        """
        rewards = []
        results = {'win': 0, 'loss': 0, 'draw': 0}
        env_params = {
            'mode': self.env.mode,
            'keep_mode': self.args.keep_mode,
            'reward': self.args.reward
        }
        max_workers = min(eval_episodes, os.cpu_count() or 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            base_seed = self.args.seed if self.args.seed is not None else 0
            for i in range(eval_episodes):
                seed = base_seed + i
                futures.append(executor.submit(self._run_eval_episode, self.agent, self.opponent, env_params, seed))
            for future in concurrent.futures.as_completed(futures):
                episode_reward, winner = future.result()
                rewards.append(episode_reward)
                if winner == 1:
                    results['win'] += 1
                elif winner == -1:
                    results['loss'] += 1
                else:
                    results['draw'] += 1

        avg_reward = np.mean(rewards) if rewards else 0
        total_games = results['win'] + results['loss'] + results['draw']
        win_ratio = results['win'] / total_games if total_games > 0 else 0
        print(f"[Parallel Eval] Win Ratio: {win_ratio:.3f}, Avg Reward: {avg_reward:.2f}")
        return win_ratio

    def _run_eval_episode(self, agent, opponent, env_params, seed):
        from hockey_env import HockeyEnv
        env = HockeyEnv(mode=env_params['mode'], keep_mode=env_params['keep_mode'], reward=env_params['reward'])
        if seed is not None:
            env.set_seed(seed)
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        trunc = False
        while not (done or trunc):
            agent_action = agent.act(obs, eval_mode=True)
            if opponent is not None:
                if hasattr(env, "obs_agent_two"):
                    opponent_obs = env.obs_agent_two()
                else:
                    opponent_obs = obs
                opponent_action = opponent.act(opponent_obs)
                full_action = np.hstack([agent_action, opponent_action])
            else:
                zeros = np.zeros(env.action_space.shape[0] // 2, dtype=np.float32)
                full_action = np.hstack([agent_action, zeros])
            obs, reward, done, trunc, info = env.step(full_action)
            total_reward += reward

        env.close()
        winner = info.get("winner", 0)
        return total_reward, winner



# =============================================================================
# Command-Line Argument Parsing and Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SAC trainer with advanced hyperparameters, Opponent Play & Win Ratio Eval"
    )
    parser.add_argument("--name", type=str, default="SAC")
    parser.add_argument("--env_name", type=str, default="Pendulum-v1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_episodes", type=int, default=2000)
    parser.add_argument("--max_timesteps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--update_every", type=float, default=1)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--buffer_size", type=int, default=1000000)
    parser.add_argument("--id", type=int, default=os.getpid())
    # PER / ERE
    parser.add_argument("--use_per", action="store_true", default=False)
    parser.add_argument("--use_ere", action="store_true", default=False)
    parser.add_argument("--per_alpha", type=float, default=0.6)
    parser.add_argument("--per_beta", type=float, default=0.4)
    parser.add_argument("--ere_eta0", type=float, default=0.996)
    parser.add_argument("--ere_min_size", type=int, default=2500)
    # Logging
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=20)
    # Noise
    parser.add_argument("--noise_type", type=str, default="colored")
    parser.add_argument("--noise_sigma", type=float, default=0.1)
    parser.add_argument("--noise_theta", type=float, default=0.15)
    parser.add_argument("--noise_dt", type=float, default=0.01)
    parser.add_argument("--noise_beta", type=float, default=1.0)
    parser.add_argument("--noise_seq_len", type=int, default=1000)
    # Hockey-specific
    parser.add_argument("--hockey_mode", type=str, default="NORMAL")
    parser.add_argument("--opponent_type", type=str, default="none")  # "none","weak"/"weak_basic","strong","trained"
    parser.add_argument("--keep_mode", action="store_true", default=False)
    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=2000, help="Eval every N episodes (0=disable)")
    parser.add_argument("--eval_episodes", type=int, default=1000, help="Number of episodes for eval")
    # Advanced hyperparameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for SAC updates")
    parser.add_argument("--hidden_sizes_actor", type=str, default="256,256", help="Comma-separated hidden layer sizes")
    parser.add_argument("--hidden_sizes_critic", type=str, default="256,256", help="Comma-separated hidden layer sizes for critic")
    parser.add_argument("--tau", type=float, default=0.005, help="Polyak averaging coefficient")
    parser.add_argument("--learn_alpha", type=str, default="true", help="True or False, whether to learn temperature")
    parser.add_argument("--alpha", type=float, default=0.2, help="Initial or fixed alpha")
    parser.add_argument("--reward", type=str, default="basic", help="Reward type: basic, middle, advanced")
    parser.add_argument("--mirror", action="store_true", default=False, help="Mirror states and actions")

    # *** SELF-PLAY ARGUMENTS ***
    parser.add_argument("--self_play", action="store_true", default=False,
                        help="If set, will use the self-play training loop.")
    parser.add_argument("--sp_min_epochs", type=int, default=500,
                        help="Number of epochs to train before checking threshold in self-play.")
    parser.add_argument("--sp_threshold", type=float, default=4.0,
                        help="Reward threshold to beat to update opponent in self-play.")
    parser.add_argument("--sp_opponent_checkpoint", type=str, default="",
                        help="Path to a checkpoint for the initial opponent (if self-play + trained).")
    parser.add_argument("--sp_opponent_config", type=str, default="",
                        help="Path to config.json for the initial opponent (if self-play + trained).")
    parser.add_argument("--sp_agent_checkpoint", type=str, default="",
                    help="Path to a checkpoint for the *agent* that can beat the strong opponent. "
                         "If provided, the trainer will load this agent before starting self-play.")
    parser.add_argument("--sp_agent_config", type=str, default="",
                    help="Path to the config.json for the above agent checkpoint.")
    parser.add_argument("--sp_switch_prob", type=float, default=0.05,
                        help="Probability of switching the opponent with an old version or the basic opponent.")

    parser.add_argument("--sp_mode", type=int, default=1,
                        help="Which self-play approach to use? 1=old approach, 2=prioritized buffer approach.")
    parser.add_argument("--sp_wr_threshold", type=float, default=0.95,
                        help="Win rate threshold for self-play mode 2.")
    parser.add_argument("--sp_n_update", type=int, default=1000,
                        help="How often to update the opponent in self-play mode 2.")
    parser.add_argument("--sp_load", action="store_true", default=False,
                        help="If set, will load the agent and opponent from checkpoints.")
    parser.add_argument("--sp_opponents_folder", type=str, default="",
                        help="Folder with SAC agents to load as opponents for mode 4.")
    
    
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(args)
    
    if args.self_play:
        if args.sp_mode == 1:
            trainer.train_self_play()
        elif args.sp_mode == 2:
            trainer.train_self_play_mode2()
        elif args.sp_mode == 3:
            trainer.train_self_play_mode3()
        elif args.sp_mode == 4:
            trainer.train_self_play_mode4()
        else:
            print(f"Unknown self_play_mode={args.sp_mode}, defaulting to mode 1.")
            trainer.train_self_play()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
