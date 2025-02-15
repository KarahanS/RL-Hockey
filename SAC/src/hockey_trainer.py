#!/usr/bin/env python
"""
SAC Trainer for Hockey Environment with Custom Classes

This script trains an agent using Soft Actor-Critic (SAC) in the hockey
environment. It uses custom classes to wrap the hockey environment to a 
Gym-like API and provides support for opponent play via simple opponent classes.
"""

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
import os


from sac import SACAgent
from memory import PrioritizedExperienceReplay
from hockey_env import HockeyEnv, BasicOpponent, Mode, BasicDefenseOpponent, BasicAttackOpponent   # your hockey environment and basic opponent
from noise import *  # your noise module
from enum import Enum

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
        # HockeyEnv.step returns (obs, reward, done, trunc, info)
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


class TrainedOpponent:
    """
    An opponent that uses a trained agent to select actions.
    """
    def __init__(self, agent, training=False):
        self.agent = agent
        self.training = training

    def act(self, observation, add_noise=False):
        observation = torch.FloatTensor(observation).to(self.agent.device)
        self.agent.actor.eval()
        with torch.no_grad():
            action = self.agent.actor(observation).cpu().numpy()
        if add_noise:
            action += self.agent.exploration_noise()
            action = np.clip(action, -self.agent.max_action, self.agent.max_action)
        self.agent.actor.train()
        return action

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
        # Determine the mode from arguments.
        mode = Mode.NORMAL
        if hasattr(self.args, "hockey_mode"):
            if isinstance(self.args.hockey_mode, str):
                mode = Mode[self.args.hockey_mode.upper()]
            else:
                mode = Mode(self.args.hockey_mode)
                
        keep_mode = self.args.keep_mode
        # Directly instantiate the environment.
        self.env = HockeyEnv(mode=mode, keep_mode=keep_mode, reward=self.args.reward)
        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            self.env.action_space.seed(self.args.seed)

    def setup_agent(self):
        # Convert comma-separated strings into lists of integers.
        hidden_actor = list(map(int, self.args.hidden_sizes_actor.split(",")))
        hidden_critic = list(map(int, self.args.hidden_sizes_critic.split(",")))
        learn_alpha_bool = self.args.learn_alpha.lower() == "true"
        #self.args.beta_frames = self.args.max_episodes * self.args.max_timesteps
        
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
        else:
            self.opponent = None
        
        if self.args.hockey_mode.lower() == "train_defense":
            self.opponent = BasicAttackOpponent(keep_mode=self.args.keep_mode)
        elif self.args.hockey_mode.lower() == "train_shooting":
            self.opponent = BasicDefenseOpponent(keep_mode=self.args.keep_mode)
            
            

    def get_run_name(self):
        if self.args.name != "SAC":
            # append id
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
        # Compute an opponent description for logging.
        opponent_desc = "None"
        if self.opponent is not None:
            # If the opponent is a BasicOpponent, distinguish between weak and strong.
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
            "\nRun Name:",
            f"  {self.run_name}",
        ]
        # Filter out any empty strings that may result from conditional lines.
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
        #weights_ckpt_path = self.output_dir / f"weights_only_episode_{episode}.pth"
        #torch.save(minimal_ckpt, weights_ckpt_path)
        
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

    def train(self):
        for episode in range(1, self.args.max_episodes + 1):
            self.agent.K = 0
            self.agent.reset_noise()
            obs, _info = self.env.reset()
            total_reward = 0.0
            
            # rollout phase
            for t in range(self.args.max_timesteps):
                self.timestep += 1
                agent_action = self.agent.act(obs, eval_mode=False, rollout=True)
                if self.opponent is not None:
                    opponent_obs = self.env.obs_agent_two() if hasattr(self.env, "obs_agent_two") else obs
                    if isinstance(self.opponent, TrainedOpponent):
                        opponent_action = self.opponent.act(opponent_obs, eval_mode=True, rollout=False) # opponent is always in eval mode
                    else:
                        opponent_action = self.opponent.act(opponent_obs)
                    full_action = np.hstack([agent_action, opponent_action])
                else:
                    # If no opponent, just use the agent's action.
                    opponent_action = np.array([0, 0, 0, 0], dtype=np.float32)
                    full_action = np.hstack([agent_action, opponent_action])
                    
                next_obs, reward, done, trunc, info = self.env.step(full_action)
                total_reward += reward
                self.agent.store_transition((obs, agent_action, reward, next_obs, done))
                
                # ----- Minimal change for symmetry: -----
                
                #mirrored_obs = self.env.mirror_state(obs)
                #mirrored_agent_action = self.env.mirror_action(agent_action)
                #mirrored_next_obs = self.env.mirror_state(next_obs)
                #self.agent.store_transition((mirrored_obs, mirrored_agent_action, reward, mirrored_next_obs, done))
            
                # ----------------------------------------
                self.agent.K += 1
                obs = next_obs
                if done or trunc:
                    break

            self.losses.extend(self.agent.train(self.agent.K))
            self.rewards.append(total_reward)
            self.lengths.append(t)
            if self.args.eval_interval > 0 and episode % self.args.eval_interval == 0:
                ratio = self.evaluate_policy(eval_episodes=self.args.eval_episodes)
                self.eval_win_ratios.append(ratio)
                self.eval_episodes_done.append(episode)
                eval_msg = f"[Eval] Episode {episode} -> Win Ratio: {ratio:.3f}"
                print(eval_msg)
                with open(self.log_file, "a") as f:
                    f.write(eval_msg + "\n")
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
        
        
    def evaluate_policy(self, eval_episodes=1000):
        """
        Runs evaluation episodes in parallel and returns the win ratio.
        (You can also return additional statistics such as average reward if desired.)
        """
        rewards = []
        results = {'win': 0, 'loss': 0, 'draw': 0}
        # Extract environment parameters from the training configuration.
        env_params = {
            'mode': self.env.mode,            # e.g., Mode.NORMAL
            'keep_mode': self.args.keep_mode,  # same as used in training
            'reward': self.args.reward         # reward type string
        }
        # Determine the maximum number of workers based on available CPUs.
        max_workers = min(eval_episodes, os.cpu_count() or 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            # Optionally, if a seed is provided, vary it per episode.
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
        """
        Run one evaluation episode in a fresh environment instance.
        
        Parameters:
        - agent: The trained SACAgent.
        - opponent: The opponent (or None).
        - env_params: A dictionary containing keys 'mode', 'keep_mode', and 'reward'
                        to instantiate a fresh HockeyEnv.
        - seed: Optional seed for the environment.
        
        Returns:
        A tuple (episode_reward, winner) where winner is 1 (agent win), -1 (agent loss), or 0 (draw).
        """
        from hockey_env import HockeyEnv  # ensure fresh import
        # Create a fresh environment instance using the same parameters.
        env = HockeyEnv(mode=env_params['mode'], keep_mode=env_params['keep_mode'], reward=env_params['reward'])
        if seed is not None:
            env.set_seed(seed)
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        trunc = False
        # Run the episode
        while not (done or trunc):
            # Get action from agent.
            agent_action = agent.act(obs, eval_mode=True)
            if opponent is not None:
                # Use opponent’s observation if available.
                if hasattr(env, "obs_agent_two"):
                    opponent_obs = env.obs_agent_two()
                else:
                    opponent_obs = obs
                opponent_action = opponent.act(opponent_obs)
                full_action = np.hstack([agent_action, opponent_action])
            else:
                # If no opponent, use zeros for the second half of the action.
                zeros = np.zeros(env.action_space.shape[0] // 2, dtype=np.float32)
                full_action = np.hstack([agent_action, zeros])
            obs, reward, done, trunc, info = env.step(full_action)
            total_reward += reward
        env.close()
        # The winner is assumed to be stored in info under key "winner"
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
    parser.add_argument("--noise_type", type=str, default="normal")
    parser.add_argument("--noise_sigma", type=float, default=0.1)
    parser.add_argument("--noise_theta", type=float, default=0.15)
    parser.add_argument("--noise_dt", type=float, default=0.01)
    parser.add_argument("--noise_beta", type=float, default=1.0)
    parser.add_argument("--noise_seq_len", type=int, default=1000)
    # Hockey-specific
    parser.add_argument("--hockey_mode", type=str, default="NORMAL")
    parser.add_argument("--opponent_type", type=str, default="none")  # Options: "none", "weak"/"weak_basic", or "strong"
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
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
