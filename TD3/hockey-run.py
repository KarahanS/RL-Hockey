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

# Import your TD3 components
from src.memory import ReplayBuffer
from src.td3 import TD3

# Import the custom Hockey environment
# Add the parent directory (one level up) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hockey.hockey_env as h_env
from importlib import reload

# Reload the hockey environment to ensure the latest version is used
reload(h_env)

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
    def __init__(self, agent, training=False):
        """
        Opponent that uses a trained TD3 agent to select actions.
        If training is True, this agent will also be trained.
        """
        self.agent = agent
        self.training = training

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
        trained_agent (TD3 or None): The trained agent to use as an opponent if opponent_type is 'trained'.

    Returns:
        An opponent instance compatible with the environment.
    """
    if opponent_type == "weak":
        return h_env.BasicOpponent(weak=True)
    elif opponent_type == "strong":
        return h_env.BasicOpponent(weak=False)
    elif opponent_type == "trained":
        if trained_agent is None:
            raise ValueError("trained_agent must be provided for 'trained' opponent type.")
        return TrainedOpponent(trained_agent, training=False)
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
    trained_agent=None  # Existing parameter
):
    """
    A single evaluation function that can handle:
      - Normal mode with strong/weak/trained opponent
      - Shooting mode (no opponent)
      - Defense mode (no opponent)

    If `opponent_type` is 'strong', 'weak', or 'trained', we create that opponent.
    If `opponent_type` is None, we do not create any built-in opponent.

    Returns a dictionary of evaluation stats:
        {
            "avg_reward": float,
            "win": int,
            "loss": int,
            "draw": int,
            "win_rate": float
        }
    """
    # Create a new environment for evaluation to avoid state carry-over
    eval_env = CustomHockeyEnv(mode=mode)
    eval_env.seed(seed)

    # Set networks to eval mode
    policy.actor.eval()
    policy.critic.eval()

    # Initialize the opponent if needed
    if opponent_type in ["strong", "weak", "trained"]:
        opponent = get_opponent(opponent_type, eval_env, trained_agent=trained_agent)
    else:
        opponent = None  # e.g., for shooting or defense

    total_rewards = []
    results = {'win': 0, 'loss': 0, 'draw': 0}

    with torch.no_grad():
        for _ in range(eval_episodes):
            state, info = eval_env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Agent action
                agent_action = policy.act(np.array(state), add_noise=False)

                if opponent is not None:
                    # Opponent action
                    opponent_obs = eval_env.env.obs_agent_two()
                    opponent_action = opponent.act(opponent_obs)
                else:
                    # No built-in opponent, so second agent does nothing
                    opponent_action = np.array([0, 0, 0, 0], dtype=np.float32)

                # Combine actions
                full_action = np.hstack([agent_action, opponent_action])

                next_state, reward, done, info = eval_env.step(full_action)
                episode_reward += reward
                state = next_state

            total_rewards.append(episode_reward)

            # Determine final outcome via `env._get_info()`
            final_info = eval_env.env._get_info()
            winner = final_info.get('winner', 0)  # 1 => agent1, -1 => agent1 loses, 0 => draw

            if winner == 1:
                results['win'] += 1
            elif winner == -1:
                results['loss'] += 1
            else:
                results['draw'] += 1

    avg_reward = np.mean(total_rewards)
    total_games = results['win'] + results['loss'] + results['draw']
    win_rate = (results['win'] / total_games) if total_games > 0 else 0.0

    # Restore networks to train mode
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
    episodes = range(1, len(loss_data_main["critic_loss"]) + 1)

    plt.figure(figsize=(14, 6))

    # Plot Critic Loss
    plt.subplot(1, 2, 1)
    plt.plot(episodes, loss_data_main["critic_loss"], label='Main Critic Loss', color='blue')
    if loss_data_opponent is not None:
        plt.plot(episodes, loss_data_opponent["critic_loss"], label='Opponent Critic Loss', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title(f'Critic Loss over Episodes for {mode}')
    plt.legend()
    plt.grid(True)

    # Plot Actor Loss
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
    Plots multiple win-rate curves in the same figure.
    `winrate_dict` should be a dictionary like:
       {
         "strong": [...],
         "weak": [...],
         "trained_1": [...],
         "trained_2": [...],
         "shooting": [...],
         "defense": [...]
       }
    Each list is the time series of the agent's win rate in that scenario.
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
    Plots Pink/OU noise vs Gaussian noise for comparison, if the agent uses pink or OU noise.
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


# ============================================
# Logging Setup
# ============================================
def setup_logging(results_dir, seed, mode):
    log_file = os.path.join(results_dir, f"training_log_seed_{seed}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            #logging.StreamHandler()  # Uncomment to also output logs to console
        ]
    )
    logging.info(f"Training started with mode: {mode}, seed: {seed}")


# ============================================
# Save Training Information
# ============================================
def save_training_info(config, mode, mixed_cycle, opponent_agent_paths, results_dir):
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
# Utility for Mixed Mode (if needed)
# ============================================
def get_trained_opponent_index(episode, num_trained_opponents):
    if num_trained_opponents == 0:
        return None
    return (episode - 1) % num_trained_opponents


# ============================================
# Training Loop
# ============================================
def main(
    mode="vs_strong",
    episodes=700,
    seed=42,
    save_model=True,
    training_config=None,
    load_agent_path=None,
    opponent_agent_paths=None  # list of paths for pre-existing opponent checkpoints (used in mixed mode)
):
    # Default training configuration if none provided
    if training_config is None:
        training_config = {
            "discount": 0.99,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_freq": 2,
            "max_episodes": episodes,
            "start_timesteps": 1000,
            "eval_freq": 100,
            "batch_size": 256,
            "expl_noise_type": "pink",
            "expl_noise": 0.1,
            "pink_noise_params": {"exponent": 1.0, "fmin": 0.0},
            "ou_noise_params": {"mu": 0.0, "theta": 0.15, "sigma": 0.2},
            "use_layer_norm": True,
            "ln_eps": 1e-5,
            "save_model": save_model,
            "save_model_freq": 10000,
            "use_rnd": True,
            "rnd_weight": 1.0,
            "rnd_lr": 1e-4,
            "rnd_hidden_dim": 128,
            "max_episode_steps": 600
        }

    # Decide environment mode based on user choice
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

    file_name = f"TD3_Hockey_{mode}_seed_{seed}"
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
            ("strong", h_env.Mode.NORMAL),
            ("weak", h_env.Mode.NORMAL),
            ("trained", h_env.Mode.NORMAL),
            (None, h_env.Mode.TRAIN_SHOOTING),
            (None, h_env.Mode.TRAIN_DEFENSE)
        ]
    else:
        mixed_cycle = None

    save_training_info(training_config, mode, mixed_cycle, opponent_agent_paths, results_dir)

    # ============================================
    # Initialize Agents and Replay Buffers
    # ============================================
    if mode == "self_play":
        current_env_mode = h_env.Mode.NORMAL
        # Initialize only the main agent
        agent = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action, training_config=training_config)
        if load_agent_path is not None:
            logging.info(f"Loading main agent from: {load_agent_path}")
            agent.load(load_agent_path)

        # For self-play, use a bank of checkpoints for the opponent.
        # Do NOT load any pre-existing checkpoints from disk in self_play mode.
        opponent_bank_dir = os.path.join(models_dir, "opponent_bank")
        os.makedirs(opponent_bank_dir, exist_ok=True)
        opponent_bank_paths = []  # the bank builds during training

        # Initialize opponent as a copy of the current agent (bank is empty initially)
        opponent_agent = copy.deepcopy(agent)
        logging.info("No opponent checkpoint available; using a copy of the main agent as the opponent.")
        opponent = TrainedOpponent(opponent_agent, training=False)

        replay_buffer_main = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
    else:
        agent = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action, training_config=training_config)
        if load_agent_path is not None:
            logging.info(f"Loading agent from: {load_agent_path}")
            agent.load(load_agent_path)
        opponent = None
        replay_buffer_main = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
        if mode in ["vs_strong", "vs_weak", "mixed"]:
            replay_buffer_opponent = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
        else:
            replay_buffer_opponent = None

    # For mixed mode, load opponent agents if provided
    opponent_agents = []
    if mode == "mixed" and opponent_agent_paths is not None:
        for idx, path in enumerate(opponent_agent_paths):
            logging.info(f"Loading opponent agent {idx + 1} from: {path}")
            trained_agent = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action, training_config=training_config)
            trained_agent.load(path)
            trained_agent.critic.eval()
            trained_agent.actor.eval()
            trained_agent.actor_target.eval()
            trained_agent.critic_target.eval()
            opponent_agents.append(trained_agent)
    elif mode == "self_play":
        pass

    total_timesteps = 0
    evaluation_results = []
    loss_results = {"critic_loss": [], "actor_loss": []}
    plot_data = {
        "evaluation_results": [],
        "loss_results": loss_results,
        "winrates_vs_scenarios": {}
    }

    # For self-play, parameters for checkpoint bank
    if mode == "self_play":
        checkpoint_freq = training_config.get("checkpoint_freq", 1000)
        max_bank_size = training_config.get("max_bank_size", 10)

    evaluation_winrates = {
        "strong": [],
        "weak": [],
        "shooting": [],
        "defense": []
    }
    if mode == "self_play":
        evaluation_winrates["against_checkpoint"] = []

    pbar = tqdm(total=training_config["max_episodes"], desc=f"Training {mode} Seed {seed}")
    for episode in range(1, training_config["max_episodes"] + 1):
        if mode == "self_play":
            # In self-play mode, use the bank of checkpoints built during training.
            if len(opponent_bank_paths) > 0 and (episode % checkpoint_freq == 1):
                rand_checkpoint = random.choice(opponent_bank_paths)
                print(f"Episode {episode}: Loading opponent from checkpoint {rand_checkpoint}")
                opponent_agent = TD3(state_dim, action_dim, max_action, training_config)
                opponent_agent.load(rand_checkpoint)
                opponent = TrainedOpponent(opponent_agent, training=False)
                logging.info(f"Episode {episode}: Loaded opponent from checkpoint {rand_checkpoint}")
            elif len(opponent_bank_paths) == 0 and (episode % checkpoint_freq == 1):
                opponent = TrainedOpponent(copy.deepcopy(agent), training=False)
                print(f"Episode {episode}: No opponent checkpoint available; using current agent copy.")
                logging.info(f"Episode {episode}: No opponent checkpoint available; using current agent copy.")
        else:
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
                    if mode == "self_play":
                        opponent = TrainedOpponent(opponent.agent, training=False)
                    else:
                        opponent = get_opponent(opponent_type, env, trained_agent=selected_trained_agent)
                else:
                    opponent = None

        if env.env.mode != current_env_mode:
            env.close()
            env = CustomHockeyEnv(mode=current_env_mode)
            env.seed(seed + episode)
            logging.info(f"Switched environment mode to: {current_env_mode}")

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
        actor_loss_steps = 0

        while not done:
            episode_timesteps += 1
            total_timesteps += 1

            if total_timesteps < training_config["start_timesteps"]:
                action = env.env.action_space.sample()[:action_dim]
            else:
                action = agent.act(np.array(state), add_noise=True)

            if opponent is not None:
                opponent_obs = env.env.obs_agent_two()
                opponent_action = opponent.act(opponent_obs, add_noise=False)
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

        if mode == "self_play" and (episode % training_config.get("checkpoint_freq", 1000) == 0):
            opponent_ckpt_path = os.path.join(os.path.join(models_dir, "opponent_bank"), f"opponent_checkpoint_{episode}.pth")
            agent.save(opponent_ckpt_path)
            opponent_bank_paths.append(opponent_ckpt_path)
            logging.info(f"Added opponent checkpoint: {opponent_ckpt_path}")
            if len(opponent_bank_paths) > training_config.get("max_bank_size", 10):
                removed = opponent_bank_paths.pop(0)
                logging.info(f"Removed oldest checkpoint from bank: {removed}")

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

            if mode == "self_play":
                stats_self = eval_policy_extended(policy=agent, eval_episodes=100, seed=seed+15, mode=h_env.Mode.NORMAL, opponent_type="trained", trained_agent=opponent.agent)
                evaluation_winrates["against_checkpoint"].append(stats_self["win_rate"])
                logging.info(f"  vs Checkpoint Opponent  => WinRate: {stats_self['win_rate']:.2f}")
                print(f"  vs Checkpoint Opponent  => WinRate: {stats_self['win_rate']:.2f}")

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
                if mode == "self_play":
                    opponent.agent.save(os.path.join(models_dir, f"{file_name}_opponent_episode_{episode}.pth"))
                logging.info(f"Saved model at episode {episode}")
                if mode == "self_play":
                    logging.info(f"Saved opponent model at episode {episode}")

        pbar.update(1)
    pbar.close()

    if save_model:
        agent.save(os.path.join(models_dir, f"{file_name}_final.pth"))
        logging.info(f"Saved final main model to {models_dir}")
        if mode == "self_play":
            opponent.agent.save(os.path.join(models_dir, f"{file_name}_opponent_final.pth"))
            logging.info(f"Saved final opponent model to {models_dir}")

    plot_losses(loss_results, None, mode, results_dir)
    plot_rewards(evaluation_results, None, mode, results_dir, window=20)
    if len(evaluation_winrates.get("strong", [])) > 0:
        plot_multi_winrate_curves(evaluation_winrates, mode, results_dir)
    plot_noise_comparison(agent, training_config, mode, results_dir)

    logging.info("\n===== Final Extended Evaluation (100 episodes each scenario) =====")
    final_stats_strong = eval_policy_extended(agent, mode=h_env.Mode.NORMAL, opponent_type="strong", seed=seed+100)
    final_stats_weak = eval_policy_extended(agent, mode=h_env.Mode.NORMAL, opponent_type="weak", seed=seed+200)
    if mode == "mixed" and len(opponent_agents) > 0:
        for idx, trained_agent in enumerate(opponent_agents):
            final_stats_trained = eval_policy_extended(agent, mode=h_env.Mode.NORMAL, opponent_type="trained", seed=seed+250+idx, trained_agent=trained_agent)
            logging.info(f"  vs Trained_{idx + 1} => WinRate: {final_stats_trained['win_rate']:.2f}")
            print(f"  vs Trained_{idx + 1} => WinRate: {final_stats_trained['win_rate']:.2f}")
    if mode == "self_play":
        final_stats_self = eval_policy_extended(agent, mode=h_env.Mode.NORMAL, opponent_type="trained", seed=seed+350, trained_agent=opponent.agent)
        logging.info(f"  vs Self    => WinRate: {final_stats_self['win_rate']:.2f}")
        print(f"  vs Self    => WinRate: {final_stats_self['win_rate']:.2f}")

    final_stats_shooting = eval_policy_extended(agent, mode=h_env.Mode.TRAIN_SHOOTING, opponent_type=None, seed=seed+300)
    final_stats_defense = eval_policy_extended(agent, mode=h_env.Mode.TRAIN_DEFENSE, opponent_type=None, seed=seed+400)

    logging.info(f"  vs Strong  => WinRate: {final_stats_strong['win_rate']:.2f}")
    logging.info(f"  vs Weak    => WinRate: {final_stats_weak['win_rate']:.2f}")
    logging.info(f"  Shooting   => WinRate: {final_stats_shooting['win_rate']:.2f}")
    logging.info(f"  Defense    => WinRate: {final_stats_defense['win_rate']:.2f}")
    if mode == "self_play":
        logging.info(f"  vs Self    => WinRate: {final_stats_self['win_rate']:.2f}")

    final_evaluation_results = {
        "final_eval_strong": final_stats_strong,
        "final_eval_weak": final_stats_weak,
        "final_eval_shooting": final_stats_shooting,
        "final_eval_defense": final_stats_defense,
    }
    if mode == "mixed" and len(opponent_agents) > 0:
        for idx, trained_agent in enumerate(opponent_agents):
            key = f"final_eval_trained_{idx + 1}"
            final_evaluation_results[key] = eval_policy_extended(agent, mode=h_env.Mode.NORMAL, opponent_type="trained", seed=seed+250+idx, trained_agent=trained_agent)
            logging.info(f"  vs Trained_{idx + 1} => WinRate: {final_evaluation_results[key]['win_rate']:.2f}")
            print(f"  vs Trained_{idx + 1} => WinRate: {final_evaluation_results[key]['win_rate']:.2f}")
    if mode == "self_play":
        final_evaluation_results["final_eval_self_play"] = final_stats_self

    final_evaluation_results.update({
        "loss_results": loss_results,
        "training_rewards": evaluation_results,
        "winrates_vs_scenarios": evaluation_winrates
    })

    if len(evaluation_winrates.get("strong", [])) > 0:
        plot_multi_winrate_curves(evaluation_winrates, mode, results_dir)

    with open(os.path.join(results_dir, f"{file_name}_final_evaluations.json"), "w") as f:
        json.dump(final_evaluation_results, f, indent=4)

    logging.info("Training completed successfully.")
    return final_evaluation_results


# ============================================
# Entry Point
# ============================================
if __name__ == "__main__":
    mode = "self_play"  # options: "vs_weak", "vs_strong", "shooting", "defense", "mixed", "self_play"
    episodes = 50000
    seed = 46
    save_model = True
    custom_training_config = {
        "discount": 0.99,
        "tau": 0.005,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "policy_freq": 2,
        "max_episodes": episodes,
        "start_timesteps": 1000,
        "eval_freq": 200,
        "batch_size": 256,
        "expl_noise_type": "pink",
        "expl_noise": 0.1,
        "pink_noise_params": {"exponent": 1.0, "fmin": 0.0},
        "ou_noise_params": {"mu": 0.0, "theta": 0.15, "sigma": 0.2},
        "use_layer_norm": True,
        "ln_eps": 1e-5,
        "save_model": save_model,
        "save_model_freq": 10000,
        "checkpoint_freq": 1000,
        "max_bank_size": 10,
        "use_rnd": True,
        "rnd_weight": 1.0,
        "rnd_lr": 1e-4,
        "rnd_hidden_dim": 128,
        "max_episode_steps": 600
    }
    # For mixed mode, you may list pre-trained opponent checkpoints.
    # In self_play mode these are not loaded.
    trained_opponent_paths = [
        "models_hockey/mixed/seed_44/TD3_Hockey_mixed_seed_44_final.pth"
    ]
    final_results = main(
        mode=mode,
        episodes=episodes,
        seed=seed,
        save_model=save_model,
        training_config=custom_training_config,
        load_agent_path="models_hockey/mixed/seed_44/TD3_Hockey_mixed_seed_44_final.pth",
        opponent_agent_paths=trained_opponent_paths
    )
    print("===== Final Results =====")
    print(json.dumps(final_results, indent=4))
