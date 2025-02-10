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
from sac import (
    SACAgent,
)

from noise import *
from enum import Enum

class Mode(Enum):
    NORMAL = 0
    TRAIN_SHOOTING = 1
    TRAIN_DEFENSE = 2

class Trainer:
    def __init__(self, args):
        self.args = args
        self.setup_environment()
        self.setup_agent()
        self.setup_logging()


    def setup_environment(self):
        """Initialize the environment and set random seeds"""
        if self.args.env_name == "LunarLander-v3":
            self.env = gym.make(self.args.env_name, continuous=True)
        else:
            self.env = gym.make(self.args.env_name)

        # Set random seeds if specified
        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            self.env.action_space.seed(self.args.seed)
    
    def setup_agent(self):
        """Initialize the SAC agent"""
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
            control_half=False   # True by default for hockey environment
        )

    def get_run_name(self):
        if self.args.name != "SAC":
            # append id
            return str(self.args.id) + "_" + self.args.name
        
        """Generate a descriptive run name based on hyperparameters"""
        components = [
            f"{self.args.id}",  # Add PID for uniqueness
            f"{self.args.env_name}",
            f"lr:{self.args.lr}",
            f"seed:{self.args.seed}" if self.args.seed is not None else "seed:none",
        ]

        # Add boolean flags if they're True
        if self.args.use_per:
            components.append(f"PER-a:{self.args.per_alpha}-b:{self.args.per_beta}")
        if self.args.use_ere:
            components.append(f"ERE-eta:{self.args.ere_eta0}")

        # Add timestamp at the end
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
        self.create_summary_file()
        np.savez(
            self.output_dir / "rewards.npz",
            rewards=np.array([]),
            episode_lengths=np.array([]),
            moving_avg_100=np.array([]),
            config=json.dumps(vars(self.args)),
        )

    def create_summary_file(self):        
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
            "\nRun Name:",
            f"  {self.run_name}",
        ]
        # Filter out any empty strings that may result from conditional lines.
        summary = [s for s in summary if s.strip()]
        with open(self.output_dir / "experiment_summary.txt", "w") as f:
            f.write("\n".join(summary))


    def save_checkpoint(self, episode):
        """Save model checkpoint and statistics"""
        checkpoint_path = self.output_dir / f"checkpoint_episode_{episode}.pth"
        torch.save(self.agent.state(), checkpoint_path)

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
        )

    def plot_metrics(self):
        """Plot and save training metrics"""
        # Plot rewards
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards)
        plt.title(f"Episode Rewards - {self.args.env_name}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.savefig(self.output_dir / "rewards.png")
        plt.close()

        # Plot episode lengths
        plt.figure(figsize=(10, 5))
        plt.plot(self.lengths)
        plt.title(f"Episode Lengths - {self.args.env_name}")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.savefig(self.output_dir / "lengths.png")
        plt.close()

        # Plot recent losses
        if self.losses:
            recent_losses = np.array(self.losses[-1000:])  # Plot last 1000 losses
            plt.figure(figsize=(10, 5))
            plt.plot(recent_losses[:, 0], label="Critic 1")
            plt.plot(recent_losses[:, 1], label="Critic 2")
            plt.plot(recent_losses[:, 2], label="Actor")
            plt.title(f"Recent Training Losses - {self.args.env_name}")
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(self.output_dir / "losses.png")
            plt.close()

        # Plot moving average of rewards
        plt.figure(figsize=(10, 5))
        window_size = 100
        moving_avg = np.convolve(
            self.rewards, np.ones(window_size) / window_size, mode="valid"
        )
        plt.plot(moving_avg)
        plt.title(
            f"Moving Average Reward (window={window_size}) - {self.args.env_name}"
        )
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.savefig(self.output_dir / "rewards_moving_avg.png")
        plt.close()

    def train(self):
        """Main training loop"""
        for episode in range(1, self.args.max_episodes + 1):
            self.agent.K = 0
            self.agent.reset_noise()
            obs, _info = self.env.reset()
            total_reward = 0

            # Episode loop
            for t in range(self.args.max_timesteps):
                self.timestep += 1

                action = self.agent.act(obs)
                next_obs, reward, done, trunc, _info = self.env.step(action)

                total_reward += reward
                self.agent.store_transition((obs, action, reward, next_obs, done))
                self.agent.K += 1

                obs = next_obs
                if done or trunc:
                    break

            # number of updates is the same as episode length K
            # source: https://arxiv.org/pdf/1906.04009
            self.losses.extend(self.agent.train(self.agent.K))
            self.rewards.append(total_reward)
            self.lengths.append(t)

            # Logging and checkpoints
            if episode % self.args.save_interval == 0:
                self.save_checkpoint(episode)
                self.plot_metrics()

            if episode % self.args.log_interval == 0:
                avg_reward = np.mean(self.rewards[-self.args.log_interval :])
                avg_length = int(np.mean(self.lengths[-self.args.log_interval :]))
                log_msg = f"Episode {episode} \t avg length: {avg_length} \t reward: {avg_reward}"
                print(log_msg)
                with open(self.log_file, "a") as f:
                    f.write(log_msg + "\n")

        # Final save
        self.save_checkpoint("final")
        self.plot_metrics()


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