#!/usr/bin/env python
"""
General Training Script for the Hockey Environment with Configurable Noise, RND, and Layer Norm

This script trains an agent in a custom hockey environment using a specified
agent class. It supports multiple modes (e.g., self_play, vs_strong, vs_weak,
shooting, defense, mixed) and allows overriding training parameters (including
noise type, RND, and layer normalization options) via command-line arguments
or a JSON configuration file.

Usage Example:
    python train_hockey.py --mode self_play --episodes 70000 --seed 47 \
        --save_model --agent_class src.td3:TD3 \
        --expl_noise_type pink --expl_noise 0.1 \
        --pink_noise_exponent 1.0 --pink_noise_fmin 0.0 \
        --use_rnd --rnd_weight 1.0 --rnd_lr 1e-4 --rnd_hidden_dim 128 \
        --use_layer_norm \
        --load_agent_path models_hockey/mixed/seed_44/TD3_Hockey_mixed_seed_44_final.pth \
        --opponent_agent_paths models_hockey/mixed/seed_44/TD3_Hockey_mixed_seed_44_final.pth
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import os
import time
import json
import logging
import random
from tqdm import tqdm
import gymnasium as gym
import sys
import importlib  # For dynamic agent class loading

# Import your ReplayBuffer and a default TD3 agent (if not using another agent)
from src.memory import ReplayBuffer
from src.td3 import TD3  # Default agent if no other is provided

# Import the custom Hockey environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hockey.hockey_env as h_env
from importlib import reload

# Reload the hockey environment to ensure the latest version is used
reload(h_env)

# ============================================
# Utility: Dynamic Agent Class Loader
# ============================================
def get_agent_class(class_path):
    """
    Dynamically imports and returns the agent class given its module path and class name.
    The expected format is "module_path:ClassName".
    
    Args:
        class_path (str): String in the format "module_path:ClassName".
        
    Returns:
        The agent class.
    """
    try:
        module_name, class_name = class_path.split(":")
    except ValueError:
        raise ValueError("Agent class must be specified in the format 'module_path:ClassName'")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

# ============================================
# Environment Wrapper
# ============================================
class CustomHockeyEnv:
    def __init__(self, mode=h_env.Mode.NORMAL, render_mode=None):
        """
        Wraps the HockeyEnv to match the Gym API more closely.
        """
        self.env = h_env.HockeyEnv(mode=mode)
        self.render_mode = render_mode

    def reset(self):
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        """
        The custom step function returns (obs, r, d, t, info).
        We convert to the standard Gym format: (obs, reward, done, info).
        """
        obs, r, d, t, info = self.env.step(action)
        done = d or t
        return obs, r, done, info

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

# ============================================
# Trained Opponent Class
# ============================================
class TrainedOpponent:
    def __init__(self, agent, training=False, name="trained_agent"):
        """
        Opponent that uses a trained agent to select actions.
        If training is True, this agent will also be trained.
        """
        self.agent = agent
        self.training = training
        self.name = name

    def act(self, observation, add_noise=False):
        """
        Selects an action based on the current observation using the trained agent.
        """
        observation = torch.FloatTensor(observation).to(self.agent.device)
        self.agent.actor.eval()
        with torch.no_grad():
            action = self.agent.actor(observation).cpu().numpy()
        if add_noise:
            action += self.agent.exploration_noise()
            action = np.clip(action, -self.agent.max_action, self.agent.max_action)
        self.agent.actor.train()
        return action

    def train_agent(self, replay_buffer, batch_size):
        """
        Train the opponent agent if training is enabled.
        """
        if self.training:
            return self.agent.train(replay_buffer, batch_size)
        return None, None

# ============================================
# Opponent Definitions
# ============================================
def get_opponent(opponent_type, env, trained_agent=None):
    """
    Returns an opponent based on the specified type ('weak', 'strong', 'trained').

    Args:
        opponent_type (str): Type of opponent ('weak', 'strong', 'trained').
        env (CustomHockeyEnv): The environment instance.
        trained_agent (Agent or None): The trained agent to use as an opponent if opponent_type is 'trained'.

    Returns:
        An opponent instance compatible with the environment.
    """
    if opponent_type == "weak":
        opp = h_env.BasicOpponent(weak=True)
        opp.name = "weak"
        return opp
    elif opponent_type == "strong":
        opp = h_env.BasicOpponent(weak=False)
        opp.name = "strong"
        return opp
    elif opponent_type == "trained":
        if trained_agent is None:
            raise ValueError("trained_agent must be provided for 'trained' opponent type.")
        return TrainedOpponent(trained_agent, training=False, name="trained_agent")
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

# ============================================
# Extended Evaluation Function
# ============================================
def eval_policy_extended(
    policy, 
    eval_episodes=100, 
    seed=42,
    mode=h_env.Mode.NORMAL,
    opponent_type=None,
    trained_agent=None
):
    """
    Evaluates the policy over a number of episodes in different scenarios.
    
    Args:
        policy: The agent's policy to evaluate.
        eval_episodes (int): Number of evaluation episodes.
        seed (int): Seed for reproducibility.
        mode: Environment mode (e.g., NORMAL, TRAIN_SHOOTING, TRAIN_DEFENSE).
        opponent_type (str or None): Opponent type ('strong', 'weak', 'trained') or None.
        trained_agent: Agent instance used when opponent_type is 'trained'.
    
    Returns:
        A dictionary with evaluation statistics including average reward and win rate.
    """
    eval_env = CustomHockeyEnv(mode=mode)
    eval_env.seed(seed)

    policy.actor.eval()
    policy.critic.eval()

    if opponent_type in ["strong", "weak", "trained"]:
        opponent = get_opponent(opponent_type, eval_env, trained_agent=trained_agent)
    else:
        opponent = None

    total_rewards = []
    results = {'win': 0, 'loss': 0, 'draw': 0}

    with torch.no_grad():
        for _ in range(eval_episodes):
            state, info = eval_env.reset()
            done = False
            episode_reward = 0

            while not done:
                agent_action = policy.act(np.array(state), add_noise=False)
                if opponent is not None:
                    if isinstance(opponent, h_env.BasicOpponent):
                        opponent_action = opponent.act(eval_env.env.obs_agent_two())
                    else:
                        opponent_obs = eval_env.env.obs_agent_two()
                        opponent_action = opponent.act(opponent_obs)
                else:
                    opponent_action = np.array([0, 0, 0, 0], dtype=np.float32)

                full_action = np.hstack([agent_action, opponent_action])
                next_state, reward, done, info = eval_env.step(full_action)
                episode_reward += reward
                state = next_state

            total_rewards.append(episode_reward)

            final_info = eval_env.env._get_info()
            winner = final_info.get('winner', 0)  # 1: win, -1: loss, 0: draw

            if winner == 1:
                results['win'] += 1
            elif winner == -1:
                results['loss'] += 1
            else:
                results['draw'] += 1

    avg_reward = np.mean(total_rewards)
    total_games = results['win'] + results['loss'] + results['draw']
    win_rate = (results['win'] / total_games) if total_games > 0 else 0.0

    policy.actor.train()
    policy.critic.train()

    eval_stats = {
        "avg_reward": float(avg_reward),
        "win": results['win'],
        "loss": results['loss'],
        "draw": results['draw'],
        "win_rate": float(win_rate)
    }

    return eval_stats

# ============================================
# Plotting Functions
# ============================================
def plot_losses(loss_data_main, loss_data_opponent, mode, save_path):
    """
    Plots critic and actor losses over episodes.
    """
    episodes = range(1, len(loss_data_main["critic_loss"]) + 1)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(episodes, loss_data_main["critic_loss"], label='Main Critic Loss', color='blue')
    if loss_data_opponent is not None:
        plt.plot(episodes, loss_data_opponent["critic_loss"], label='Opponent Critic Loss', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title(f'Critic Loss over Episodes for {mode}')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(episodes, loss_data_main["actor_loss"], label='Main Actor Loss', color='orange')
    if loss_data_opponent is not None:
        plt.plot(episodes, loss_data_opponent["actor_loss"], label='Opponent Actor Loss', color='purple')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title(f'Actor Loss over Episodes for {mode}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"losses_{mode}.png"))
    plt.close()

def plot_rewards(rewards_main, rewards_opponent, mode, save_path, window=20):
    """
    Plots episode rewards and their moving average.
    """
    plt.figure(figsize=(12,6))
    plt.plot(rewards_main, label='Main Episode Reward', alpha=0.3)
    if rewards_opponent is not None:
        plt.plot(rewards_opponent, label='Opponent Episode Reward', alpha=0.3)
    if len(rewards_main) >= window:
        moving_avg_main = np.convolve(rewards_main, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards_main)), moving_avg_main, label=f'Main {window}-Episode Moving Average', color='blue')
    if rewards_opponent is not None and len(rewards_opponent) >= window:
        moving_avg_opp = np.convolve(rewards_opponent, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards_opponent)), moving_avg_opp, label=f'Opponent {window}-Episode Moving Average', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'TD3 on {mode}: Episode Rewards with Moving Average')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f"rewards_{mode}.png"))
    plt.close()

def plot_multi_winrate_curves(winrate_dict, mode, save_path):
    """
    Plots multiple win-rate curves for different scenarios.
    """
    plt.figure(figsize=(12,6))
    for scenario, wr_list in winrate_dict.items():
        if len(wr_list) > 0:
            plt.plot(wr_list, label=f'Win Rate vs {scenario.capitalize()}')
    plt.xlabel('Evaluation Index (every eval_freq episodes)')
    plt.ylabel('Win Rate')
    plt.title(f'{mode} - Win Rates Over Training (All Scenarios)')
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, f"winrates_{mode}.png"))
    plt.close()

def plot_noise_comparison(agent, training_config, env_name, save_path):
    """
    Plots a comparison between Pink/OU noise and Gaussian noise.
    """
    if training_config["expl_noise_type"].lower() not in ["pink", "ou"]:
        print(f"No noise comparison plot available for noise type: {training_config['expl_noise_type']}")
        return

    max_steps = training_config.get("max_episode_steps", 600)
    action_dim = agent.action_dim

    noise_sequence = []
    for _ in range(max_steps):
        if training_config["expl_noise_type"].lower() == "pink":
            noise = agent.pink_noise.get_noise() * training_config["expl_noise"]
        elif training_config["expl_noise_type"].lower() == "ou":
            noise = agent.ou_noise.sample() * training_config["expl_noise"]
        noise_sequence.append(noise)
    noise_sequence = np.array(noise_sequence)

    gaussian_noise_sequence = np.random.normal(0, training_config["expl_noise"], size=(max_steps, action_dim))

    for dim in range(action_dim):
        plt.figure(figsize=(12, 4))
        plt.plot(noise_sequence[:, dim], label=f'{training_config["expl_noise_type"].capitalize()} Noise')
        plt.plot(gaussian_noise_sequence[:, dim], label='Gaussian Noise', alpha=0.7)
        plt.title(f'Noise Comparison for Action Dimension {dim} in {env_name}')
        plt.xlabel('Step')
        plt.ylabel('Noise Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f"noise_comparison_dim_{dim}_{env_name}.png"))
        plt.close()

def plot_selfplay_opponent_winrates(sp_winrate_history, save_path, mode):
    """
    Plots the self-play win rates for each opponent in the opponent buffer over training blocks.
    
    sp_winrate_history: Dict of {opponent_name -> [list_of_winrates]}
    Each index in the lists corresponds to a training block (50 episodes).
    """
    plt.figure(figsize=(10, 6))
    for opp_name, winrates in sp_winrate_history.items():
        plt.plot(winrates, label=f"Winrate vs {opp_name}")
    plt.xlabel("Block Index (each block ~ 50 episodes)")
    plt.ylabel("Win Rate")
    plt.title(f"Self-Play Win Rates Over Blocks - {mode}")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    out_file = os.path.join(save_path, f"selfplay_opponent_winrates_{mode}.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved self-play opponent winrates plot to {out_file}")

# ============================================
# Logging Setup
# ============================================
def setup_logging(results_dir, seed, mode):
    """
    Configures logging to file and optionally to console.
    """
    log_file = os.path.join(results_dir, f"training_log_seed_{seed}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            # Uncomment the next line to also log to console:
            # logging.StreamHandler()
        ]
    )
    logging.info(f"Training started with mode: {mode}, seed: {seed}")

# ============================================
# Save Training Information
# ============================================
def save_training_info(config, mode, mixed_cycle, opponent_agent_paths, results_dir):
    """
    Saves training configuration and settings to a JSON file.
    """
    if mixed_cycle is not None:
        mixed_cycle_serializable = []
        for opponent, mode_enum in mixed_cycle:
            mode_str = mode_enum.name if mode_enum else None
            mixed_cycle_serializable.append((opponent, mode_str))
    else:
        mixed_cycle_serializable = None

    training_info = {
        "training_config": config,
        "training_mode": mode,
        "mixed_cycle": mixed_cycle_serializable,
        "opponent_agent_paths": opponent_agent_paths
    }
    with open(os.path.join(results_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=4)

# ============================================
# Utility for Mixed Mode
# ============================================
def get_trained_opponent_index(episode, num_trained_opponents):
    """
    Returns the index of the trained opponent based on the current episode.
    """
    if num_trained_opponents == 0:
        return None
    return (episode - 1) % num_trained_opponents

# ============================================
# Main Training Loop
# ============================================
def main(
    agent_class,  # Agent class to be used for training
    mode="vs_strong",
    episodes=700,
    seed=42,
    save_model=True,
    training_config=None,
    load_agent_path=None,
    opponent_agent_paths=None
):
    """
    Main function for training the agent in various modes.
    
    Args:
        agent_class: The class of the agent to be used (must implement required methods).
        mode (str): Training mode ('vs_strong', 'vs_weak', 'shooting', 'defense', 'mixed', 'self_play').
        episodes (int): Number of training episodes.
        seed (int): Random seed.
        save_model (bool): Whether to save the model checkpoints.
        training_config (dict): Dictionary of training configuration parameters.
        load_agent_path (str or None): Path to a pre-trained agent checkpoint.
        opponent_agent_paths (list): List of paths for pre-existing opponent checkpoints (for mixed mode).
    
    Returns:
        final_evaluation_results (dict): Dictionary containing final evaluation statistics and training logs.
    """

    # Determine environment mode based on provided mode argument
    if mode == "shooting":
        env_mode = h_env.Mode.TRAIN_SHOOTING
    elif mode == "defense":
        env_mode = h_env.Mode.TRAIN_DEFENSE
    else:
        env_mode = h_env.Mode.NORMAL

    env = CustomHockeyEnv(mode=env_mode)
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = 4  # controlling first 4 actions
    max_action = float(env.action_space.high[0])

    base_results_dir = "./results_hockey"
    base_models_dir = "./models_hockey"
    os.makedirs(base_results_dir, exist_ok=True)
    os.makedirs(base_models_dir, exist_ok=True)

    file_name = f"{agent_class.__name__}_Hockey_{mode}_seed_{seed}"
    results_dir = os.path.join(base_results_dir, mode, f"seed_{seed}")
    models_dir = os.path.join(base_models_dir, mode, f"seed_{seed}")
    os.makedirs(results_dir, exist_ok=True)
    if save_model:
        os.makedirs(models_dir, exist_ok=True)

    setup_logging(results_dir, seed, mode)
    logging.info(f"Training Mode: {mode}")
    logging.info(f"Environment Mode: {env_mode}")
    logging.info(f"State Dim: {state_dim}, Action Dim: {action_dim}, Max Action: {max_action}")

    if mode == "mixed":
        mixed_cycle = [
            ("strong", h_env.Mode.NORMAL),
            ("strong", h_env.Mode.NORMAL),
            ("weak", h_env.Mode.NORMAL),
            (None, h_env.Mode.TRAIN_SHOOTING),
            (None, h_env.Mode.TRAIN_DEFENSE)
        ]
    else:
        mixed_cycle = None

    save_training_info(training_config, mode, mixed_cycle, opponent_agent_paths, results_dir)

    # Initialize agent
    agent = agent_class(state_dim=state_dim, action_dim=action_dim, max_action=max_action, training_config=training_config)
    if load_agent_path is not None:
        logging.info(f"Loading agent from: {load_agent_path}")
        agent.load(load_agent_path)

    # Replay Buffer for the main agent
    replay_buffer_main = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))

    # For storing training results
    total_timesteps = 0
    evaluation_results = []
    loss_results = {"critic_loss": [], "actor_loss": []}
    plot_data = {
        "evaluation_results": [],
        "loss_results": loss_results,
        "winrates_vs_scenarios": {}
    }

    evaluation_winrates = {
        "strong": [],
        "weak": [],
        "shooting": [],
        "defense": []
    }

    # If "mixed" mode, load possible trained opponents
    opponent_agents = []
    if mode == "mixed" and opponent_agent_paths:
        for idx, path in enumerate(opponent_agent_paths):
            logging.info(f"Loading opponent agent {idx + 1} from: {path}")
            trained_agent = agent_class(state_dim=state_dim, action_dim=action_dim, max_action=max_action, training_config=training_config)
            trained_agent.load(path)
            trained_agent.critic.eval()
            trained_agent.actor.eval()
            trained_agent.actor_target.eval()
            trained_agent.critic_target.eval()
            opponent_agents.append(trained_agent)

    pbar = tqdm(total=episodes, desc=f"Training {mode} Seed {seed}")

    # ==============================================
    #   SELF-PLAY LOGIC WITH BLOCKS & WINRATE PLOTS
    # ==============================================
    if mode == "self_play":
        from collections import namedtuple
        OpponentEntry = namedtuple("OpponentEntry", ["opponent", "name"])

        # Opponent buffer
        opponent_buffer = []

        # Start with "weak"
        weak_opp = get_opponent("weak", env)
        opponent_buffer.append(OpponentEntry(opponent=weak_opp, name="weak"))
        strong_added = False

        # For storing block-based self-play WR
        sp_winrate_history = {}  # {opp_name -> [list_of_winrates_per_block]}

        def evaluate_vs_all_opponents(agent, buffer):
            """
            Evaluate the agent vs each opponent in 'buffer' for a certain number of episodes.
            Return (average_win_rate, {opponent_name: opponent_wr}).
            """
            episodes_for_eval = 100
            total_win_rates = 0.0
            opp_win_rates = {}
            count = 0

            for opp_entry in buffer:
                if isinstance(opp_entry.opponent, h_env.BasicOpponent):
                    # "weak" or "strong"
                    opp_name = opp_entry.name
                    stats = eval_policy_extended(
                        policy=agent,
                        eval_episodes=episodes_for_eval,
                        seed=seed + random.randint(0,10000),
                        mode=h_env.Mode.NORMAL,
                        opponent_type=opp_name
                    )
                else:
                    # "trained" -> a self copy
                    stats = eval_policy_extended(
                        policy=agent,
                        eval_episodes=episodes_for_eval,
                        seed=seed + random.randint(0,10000),
                        mode=h_env.Mode.NORMAL,
                        opponent_type="trained",
                        trained_agent=opp_entry.opponent.agent
                    )
                wr = stats["win_rate"]
                opp_win_rates[opp_entry.name] = wr
                total_win_rates += wr
                count += 1

            average_win_rate = total_win_rates / max(count, 1)
            return average_win_rate, opp_win_rates

        def pick_lowest_winrate_opponent(buffer, opp_win_rates):
            """
            Pick from the buffer the opponent that yields the lowest WR for the agent.
            """
            min_wr = float("inf")
            min_name = None
            for opp_entry in buffer:
                name = opp_entry.name
                if opp_win_rates[name] < min_wr:
                    min_wr = opp_win_rates[name]
                    min_name = name
            # Return the OpponentEntry that corresponds to min_name
            for opp_entry in buffer:
                if opp_entry.name == min_name:
                    return opp_entry
            return buffer[0]  # fallback

        block_size = 50
        current_episode = 0

        while current_episode < episodes:
            # Evaluate vs all opponents
            avg_wr, opp_wr_dict = evaluate_vs_all_opponents(agent, opponent_buffer)
            logging.info(f"Evaluation vs Opponents at episode {current_episode}: {opp_wr_dict}, avg = {avg_wr:.2f}")

            # Record these winrates in sp_winrate_history
            # Each opponent's WR is appended as a new data point for the block
            for opp_name, wr in opp_wr_dict.items():
                sp_winrate_history.setdefault(opp_name, []).append(wr)

            # Condition to add strong
            if (not strong_added) and avg_wr > 0.80:
                strong_opp = get_opponent("strong", env)
                opponent_buffer.append(OpponentEntry(opponent=strong_opp, name="strong"))
                strong_added = True
                logging.warning("WARNING: Added 'strong' opponent to the buffer!")

            # Condition to add self-copy (if strong is already there)
            if strong_added and avg_wr > 0.80:
                # add a copy of the current agent
                copy_of_agent = copy.deepcopy(agent)
                new_opp_name = f"self_copy_{current_episode}"
                new_opp = TrainedOpponent(copy_of_agent, training=False, name=new_opp_name)
                opponent_buffer.append(OpponentEntry(opponent=new_opp, name=new_opp_name))
                logging.warning(f"WARNING: Added a new self-copy opponent: {new_opp_name}")

            # Re-evaluate to pick the lowest-WR opponent
            avg_wr, opp_wr_dict = evaluate_vs_all_opponents(agent, opponent_buffer)
            # Again record in sp_winrate_history for consistency
            for opp_name, wr in opp_wr_dict.items():
                sp_winrate_history.setdefault(opp_name, []).append(wr)

            chosen_opp_entry = pick_lowest_winrate_opponent(opponent_buffer, opp_wr_dict)
            logging.info(f"Chosen Opponent for next {block_size} episodes: {chosen_opp_entry.name} (WR={opp_wr_dict[chosen_opp_entry.name]:.2f})")

            # Train for the next block_size episodes
            block_count = 0
            while block_count < block_size and current_episode < episodes:
                if env.env.mode != h_env.Mode.NORMAL:
                    env.close()
                    env = CustomHockeyEnv(mode=h_env.Mode.NORMAL)
                    env.seed(seed + current_episode)

                state, info = env.reset()
                done = False
                episode_reward_main = 0
                episode_timesteps = 0

                if training_config["expl_noise_type"].lower() == "ou" and hasattr(agent, 'ou_noise'):
                    agent.ou_noise.reset()
                if training_config["expl_noise_type"].lower() == "pink" and hasattr(agent, 'pink_noise'):
                    agent.pink_noise.reset()

                cumulative_critic_loss = 0.0
                cumulative_actor_loss = 0.0
                loss_steps = 0

                while not done:
                    episode_timesteps += 1
                    total_timesteps += 1

                    if total_timesteps < training_config["start_timesteps"]:
                        action = env.env.action_space.sample()[:action_dim]
                    else:
                        action = agent.act(np.array(state))

                    # Opponent action
                    if isinstance(chosen_opp_entry.opponent, h_env.BasicOpponent):
                        opponent_action = chosen_opp_entry.opponent.act(env.env.obs_agent_two())
                    else:
                        opponent_obs = env.env.obs_agent_two()
                        opponent_action = chosen_opp_entry.opponent.act(opponent_obs)

                    full_action = np.hstack([action, opponent_action])
                    next_state, reward, done, info = env.step(full_action)
                    done_bool = float(done) if episode_timesteps < training_config["max_episode_steps"] else 0

                    replay_buffer_main.add(state, action, next_state, reward, done_bool)
                    state = next_state
                    episode_reward_main += reward

                    # Train main agent
                    if total_timesteps >= training_config["start_timesteps"]:
                        critic_loss, actor_loss = agent.train(replay_buffer_main, training_config["batch_size"])
                        cumulative_critic_loss += critic_loss
                        if actor_loss is not None:
                            cumulative_actor_loss += actor_loss
                            loss_steps += 1

                avg_critic_loss = cumulative_critic_loss / loss_steps if loss_steps > 0 else 0
                avg_actor_loss = cumulative_actor_loss / loss_steps if loss_steps > 0 else 0
                loss_results["critic_loss"].append(avg_critic_loss)
                loss_results["actor_loss"].append(avg_actor_loss)
                evaluation_results.append(episode_reward_main)
                plot_data["evaluation_results"].append(episode_reward_main)
                logging.info(f"[Episode {current_episode+1}] Reward: {episode_reward_main:.2f} | "
                             f"Opponent: {chosen_opp_entry.name} | "
                             f"Critic Loss: {avg_critic_loss:.4f} | Actor Loss: {avg_actor_loss:.4f}")

                block_count += 1
                current_episode += 1
                pbar.update(1)

                # Intermediate evaluations / save
                if (current_episode % training_config["eval_freq"] == 0) or (current_episode == episodes):
                    logging.info(f"===== Intermediate Evaluation at Episode {current_episode} =====")
                    stats_strong = eval_policy_extended(policy=agent, eval_episodes=30, seed=seed+10,
                                                        mode=h_env.Mode.NORMAL, opponent_type="strong")
                    evaluation_winrates["strong"].append(stats_strong["win_rate"])
                    logging.info(f"  vs Strong  => WinRate: {stats_strong['win_rate']:.2f}")

                    stats_weak = eval_policy_extended(policy=agent, eval_episodes=30, seed=seed+20,
                                                      mode=h_env.Mode.NORMAL, opponent_type="weak")
                    evaluation_winrates["weak"].append(stats_weak["win_rate"])
                    logging.info(f"  vs Weak    => WinRate: {stats_weak['win_rate']:.2f}")

                    stats_shooting = eval_policy_extended(policy=agent, eval_episodes=30, seed=seed+30,
                                                          mode=h_env.Mode.TRAIN_SHOOTING, opponent_type=None)
                    evaluation_winrates["shooting"].append(stats_shooting["win_rate"])
                    logging.info(f"  Shooting   => WinRate: {stats_shooting['win_rate']:.2f}")

                    stats_defense = eval_policy_extended(policy=agent, eval_episodes=30, seed=seed+40,
                                                         mode=h_env.Mode.TRAIN_DEFENSE, opponent_type=None)
                    evaluation_winrates["defense"].append(stats_defense["win_rate"])
                    logging.info(f"  Defense    => WinRate: {stats_defense['win_rate']:.2f}")

                    np.save(os.path.join(results_dir, f"{file_name}_evaluations.npy"), np.array(evaluation_results))
                    np.save(os.path.join(results_dir, f"{file_name}_winrates.npy"), evaluation_winrates)
                    with open(os.path.join(results_dir, f"{file_name}_plot_data.json"), "w") as f:
                        json.dump(plot_data, f, indent=4)

                    if save_model and (current_episode % training_config["save_model_freq"] == 0):
                        agent.save(os.path.join(models_dir, f"{file_name}_episode_{current_episode}.pth"))
                        logging.info(f"Saved model at episode {current_episode}")

        # Done with all episodes in self-play
        pbar.close()

        # After finishing, let's plot the self-play WRs
        plot_selfplay_opponent_winrates(sp_winrate_history, results_dir, mode)

    # ================================
    # Other modes (unchanged logic)
    # ================================
    else:
        if mode in ["vs_strong", "vs_weak", "mixed"]:
            replay_buffer_opponent = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
        else:
            replay_buffer_opponent = None

        opponent = None
        if mode == "mixed" and opponent_agent_paths:
            for idx, path in enumerate(opponent_agent_paths):
                logging.info(f"Loading opponent agent {idx + 1} from: {path}")
                trained_agent = agent_class(state_dim=state_dim, action_dim=action_dim, max_action=max_action, training_config=training_config)
                trained_agent.load(path)
                trained_agent.critic.eval()
                trained_agent.actor.eval()
                trained_agent.actor_target.eval()
                trained_agent.critic_target.eval()
                opponent_agents.append(trained_agent)

        for episode in range(1, episodes + 1):
            if mode in ["vs_strong", "vs_weak", "mixed"]:
                if mode == "mixed":
                    idx = (episode - 1) % len(mixed_cycle)
                    cycle_opponent_type, cycle_env_mode = mixed_cycle[idx]
                    if cycle_opponent_type == "trained" and len(opponent_agents) > 0:
                        trained_idx = get_trained_opponent_index(episode, len(opponent_agents))
                        opponent_type = "trained"
                        current_env_mode = cycle_env_mode
                        selected_trained_agent = opponent_agents[trained_idx]
                    else:
                        opponent_type = cycle_opponent_type
                        current_env_mode = cycle_env_mode
                        selected_trained_agent = None
                else:
                    opponent_type = "strong" if mode == "vs_strong" else "weak"
                    current_env_mode = env_mode
                    selected_trained_agent = None

                if opponent_type in ["strong", "weak", "trained"]:
                    opponent = get_opponent(opponent_type, env, trained_agent=selected_trained_agent)
                else:
                    opponent = None

                if env.env.mode != current_env_mode:
                    env.close()
                    env = CustomHockeyEnv(mode=current_env_mode)
                    env.seed(seed + episode)

            state, info = env.reset()
            done = False
            episode_reward_main = 0
            episode_timesteps = 0

            if training_config["expl_noise_type"].lower() == "ou" and hasattr(agent, 'ou_noise'):
                agent.ou_noise.reset()
            if training_config["expl_noise_type"].lower() == "pink" and hasattr(agent, 'pink_noise'):
                agent.pink_noise.reset()

            cumulative_critic_loss = 0.0
            cumulative_actor_loss = 0.0
            loss_steps = 0

            while not done:
                episode_timesteps += 1
                total_timesteps += 1

                if total_timesteps < training_config["start_timesteps"]:
                    action = env.env.action_space.sample()[:action_dim]
                else:
                    action = agent.act(np.array(state))

                if opponent is not None:
                    if isinstance(opponent, h_env.BasicOpponent):
                        opponent_action = opponent.act(env.env.obs_agent_two())
                    else:
                        opponent_obs = env.env.obs_agent_two()
                        opponent_action = opponent.act(opponent_obs)
                else:
                    opponent_action = np.array([0, 0, 0, 0], dtype=np.float32)

                full_action = np.hstack([action, opponent_action])
                next_state, reward, done, info = env.step(full_action)
                done_bool = float(done) if episode_timesteps < training_config["max_episode_steps"] else 0

                replay_buffer_main.add(state, action, next_state, reward, done_bool)
                state = next_state
                episode_reward_main += reward

                if total_timesteps >= training_config["start_timesteps"]:
                    critic_loss, actor_loss = agent.train(replay_buffer_main, training_config["batch_size"])
                    cumulative_critic_loss += critic_loss
                    if actor_loss is not None:
                        cumulative_actor_loss += actor_loss
                        loss_steps += 1

            avg_critic_loss = cumulative_critic_loss / loss_steps if loss_steps > 0 else 0
            avg_actor_loss = cumulative_actor_loss / loss_steps if loss_steps > 0 else 0
            loss_results["critic_loss"].append(avg_critic_loss)
            loss_results["actor_loss"].append(avg_actor_loss)
            evaluation_results.append(episode_reward_main)
            plot_data["evaluation_results"].append(episode_reward_main)
            logging.info(f"Episode {episode} | Reward: {episode_reward_main:.2f} | Critic Loss: {avg_critic_loss:.4f} | Actor Loss: {avg_actor_loss:.4f}")

            if episode % training_config["eval_freq"] == 0:
                logging.info(f"===== Evaluation at Episode {episode} =====")
                stats_strong = eval_policy_extended(policy=agent, eval_episodes=100, seed=seed+10, mode=h_env.Mode.NORMAL, opponent_type="strong")
                evaluation_winrates["strong"].append(stats_strong["win_rate"])
                logging.info(f"  vs Strong  => WinRate: {stats_strong['win_rate']:.2f}")
                print(f"  vs Strong  => WinRate: {stats_strong['win_rate']:.2f}")

                stats_weak = eval_policy_extended(policy=agent, eval_episodes=100, seed=seed+20, mode=h_env.Mode.NORMAL, opponent_type="weak")
                evaluation_winrates["weak"].append(stats_weak["win_rate"])
                logging.info(f"  vs Weak    => WinRate: {stats_weak['win_rate']:.2f}")
                print(f"  vs Weak    => WinRate: {stats_weak['win_rate']:.2f}")

                stats_shooting = eval_policy_extended(policy=agent, eval_episodes=100, seed=seed+30, mode=h_env.Mode.TRAIN_SHOOTING, opponent_type=None)
                evaluation_winrates["shooting"].append(stats_shooting["win_rate"])
                logging.info(f"  Shooting   => WinRate: {stats_shooting['win_rate']:.2f}")
                print(f"  Shooting   => WinRate: {stats_shooting['win_rate']:.2f}")

                stats_defense = eval_policy_extended(policy=agent, eval_episodes=100, seed=seed+40, mode=h_env.Mode.TRAIN_DEFENSE, opponent_type=None)
                evaluation_winrates["defense"].append(stats_defense["win_rate"])
                logging.info(f"  Defense    => WinRate: {stats_defense['win_rate']:.2f}")
                print(f"  Defense    => WinRate: {stats_defense['win_rate']:.2f}")

                np.save(os.path.join(results_dir, f"{file_name}_evaluations.npy"), np.array(evaluation_results))
                np.save(os.path.join(results_dir, f"{file_name}_winrates.npy"), evaluation_winrates)
                with open(os.path.join(results_dir, f"{file_name}_plot_data.json"), "w") as f:
                    json.dump(plot_data, f, indent=4)

                if save_model and (episode % training_config["save_model_freq"] == 0):
                    agent.save(os.path.join(models_dir, f"{file_name}_episode_{episode}.pth"))
                    logging.info(f"Saved model at episode {episode}")

        pbar.close()

    # Save final model
    if save_model:
        agent.save(os.path.join(models_dir, f"{file_name}_final.pth"))
        logging.info(f"Saved final main model to {models_dir}")

    # Plot losses and rewards
    plot_losses(loss_results, None, mode, results_dir)
    plot_rewards(evaluation_results, None, mode, results_dir, window=20)
    if len(evaluation_winrates.get("strong", [])) > 0:
        plot_multi_winrate_curves(evaluation_winrates, mode, results_dir)
    plot_noise_comparison(agent, training_config, mode, results_dir)

    # Final extended evaluation
    logging.info("\n===== Final Extended Evaluation (100 episodes each scenario) =====")
    final_stats_strong = eval_policy_extended(agent, mode=h_env.Mode.NORMAL, opponent_type="strong", seed=seed+100)
    final_stats_weak = eval_policy_extended(agent, mode=h_env.Mode.NORMAL, opponent_type="weak", seed=seed+200)
    final_stats_shooting = eval_policy_extended(agent, mode=h_env.Mode.TRAIN_SHOOTING, opponent_type=None, seed=seed+300)
    final_stats_defense = eval_policy_extended(agent, mode=h_env.Mode.TRAIN_DEFENSE, opponent_type=None, seed=seed+400)

    logging.info(f"  vs Strong  => WinRate: {final_stats_strong['win_rate']:.2f}")
    logging.info(f"  vs Weak    => WinRate: {final_stats_weak['win_rate']:.2f}")
    logging.info(f"  Shooting   => WinRate: {final_stats_shooting['win_rate']:.2f}")
    logging.info(f"  Defense    => WinRate: {final_stats_defense['win_rate']:.2f}")

    final_evaluation_results = {
        "final_eval_strong": final_stats_strong,
        "final_eval_weak": final_stats_weak,
        "final_eval_shooting": final_stats_shooting,
        "final_eval_defense": final_stats_defense,
        "loss_results": loss_results,
        "training_rewards": evaluation_results,
        "winrates_vs_scenarios": evaluation_winrates
    }

    # If mixed mode with loaded opponents, evaluate vs each
    if mode == "mixed" and len(opponent_agents) > 0:
        for idx, trained_agent in enumerate(opponent_agents):
            key = f"final_eval_trained_{idx + 1}"
            final_evaluation_results[key] = eval_policy_extended(agent, mode=h_env.Mode.NORMAL,
                                                                 opponent_type="trained",
                                                                 seed=seed+250+idx,
                                                                 trained_agent=trained_agent)
            logging.info(f"  vs Trained_{idx + 1} => WinRate: {final_evaluation_results[key]['win_rate']:.2f}")

    with open(os.path.join(results_dir, f"{file_name}_final_evaluations.json"), "w") as f:
        json.dump(final_evaluation_results, f, indent=4)

    logging.info("Training completed successfully.")
    return final_evaluation_results

# ============================================
# Command-line Argument Parsing and Entry Point
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General Training Script for Hockey Environment with Custom Agents and Configurable Noise/RND/LayerNorm")
    parser.add_argument("--mode", type=str, default="self_play",
                        help="Training mode: vs_strong, vs_weak, shooting, defense, mixed, self_play")
    parser.add_argument("--episodes", type=int, default=70000, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=47, help="Random seed for training")
    parser.add_argument("--save_model", action="store_true", help="Flag to save model checkpoints")
    parser.add_argument("--agent_class", type=str, default="src.td3:TD3",
                        help="Agent class in the format 'module_path:ClassName'. Default is 'src.td3:TD3'")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Path to JSON file containing training configuration parameters")
    parser.add_argument("--load_agent_path", type=str, default=None, help="Path to load a pre-trained agent checkpoint")
    parser.add_argument("--opponent_agent_paths", type=str, nargs="*", default=None,
                        help="List of paths for opponent agent checkpoints (for mixed mode)")

    # New arguments for configuring exploration noise
    parser.add_argument("--expl_noise_type", type=str, choices=["pink", "ou", "none", "gaussian"], default="pink",
                        help="Exploration noise type: pink, ou, gaussian or none (to disable noise)")
    parser.add_argument("--expl_noise", type=float, default=0.1, help="Exploration noise scale")
    parser.add_argument("--pink_noise_exponent", type=float, default=1.0, help="Exponent for pink noise (if used)")
    parser.add_argument("--pink_noise_fmin", type=float, default=0.0, help="Minimum frequency for pink noise (if used)")

    # New arguments for configuring RND
    group_rnd = parser.add_mutually_exclusive_group()
    group_rnd.add_argument("--use_rnd", dest="use_rnd", action="store_true", help="Enable RND (Random Network Distillation)")
    group_rnd.add_argument("--no_rnd", dest="use_rnd", action="store_false", help="Disable RND")
    parser.set_defaults(use_rnd=True)
    parser.add_argument("--rnd_weight", type=float, default=1.0, help="Weight for RND reward")
    parser.add_argument("--rnd_lr", type=float, default=1e-4, help="Learning rate for RND")
    parser.add_argument("--rnd_hidden_dim", type=int, default=128, help="Hidden dimension for RND network")

    # New arguments for configuring layer normalization
    group_ln = parser.add_mutually_exclusive_group()
    group_ln.add_argument("--use_layer_norm", dest="use_layer_norm", action="store_true", help="Enable layer normalization")
    group_ln.add_argument("--no_layer_norm", dest="use_layer_norm", action="store_false", help="Disable layer normalization")
    parser.set_defaults(use_layer_norm=True)
    parser.add_argument("--ln_eps", type=float, default=1e-5, help="Epsilon for layer normalization")

    args = parser.parse_args()

    # Load training configuration from file if provided, otherwise create an empty dictionary
    if args.config_file is not None:
        with open(args.config_file, "r") as f:
            custom_training_config = json.load(f)
    else:
        custom_training_config = {}

    training_config = {
        "discount": 0.99,
        "tau": 0.005,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "policy_freq": 2,
        "start_timesteps": 1000,
        "eval_freq": 100,
        "batch_size": 2048,
        "expl_noise_type": "pink",
        "expl_noise": 0.1,
        "pink_noise_params": {"exponent": 1.0, "fmin": 0.0},
        "ou_noise_params": {"mu": 0.0, "theta": 0.15, "sigma": 0.2},
        "use_layer_norm": True,
        "ln_eps": 1e-5,
        "save_model": True,
        "save_model_freq": 10000,
        "use_rnd": True,
        "rnd_weight": 1.0,
        "rnd_lr": 1e-4,
        "rnd_hidden_dim": 128,
        "max_episode_steps": 600,
        # The old self-play parameters are superseded by our new block-based approach
        "update_eval_interval": 200,
        "update_eval_episodes": 50,
        "update_accuracy_threshold": 0.8
    }

    # Override training_config with custom JSON config (if any)
    training_config.update(custom_training_config)

    # Override with command-line arguments for noise, RND, layer norm
    training_config["expl_noise_type"] = args.expl_noise_type
    training_config["expl_noise"] = args.expl_noise
    training_config["pink_noise_params"] = {
        "exponent": args.pink_noise_exponent,
        "fmin": args.pink_noise_fmin
    }
    training_config["use_rnd"] = args.use_rnd
    training_config["rnd_weight"] = args.rnd_weight
    training_config["rnd_lr"] = args.rnd_lr
    training_config["rnd_hidden_dim"] = args.rnd_hidden_dim
    training_config["use_layer_norm"] = args.use_layer_norm
    training_config["ln_eps"] = args.ln_eps

    print("===== Training Configuration =====")
    print(json.dumps(training_config, indent=4))

    # Dynamically load the agent class
    agent_cls = get_agent_class(args.agent_class)

    final_results = main(
        agent_class=agent_cls,
        mode=args.mode,
        episodes=args.episodes,
        seed=args.seed,
        save_model=args.save_model,
        training_config=training_config,
        load_agent_path=args.load_agent_path,
        opponent_agent_paths=args.opponent_agent_paths
    )
    print("===== Final Results =====")
    print(json.dumps(final_results, indent=4))
