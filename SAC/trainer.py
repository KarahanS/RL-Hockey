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
    select_loss_function,
)
from noise import *


class Trainer:
    def __init__(self, args):
        self.args = args
        self.setup_environment()
        self.setup_agent()
        self.setup_logging()

    def setup_environment(self):
        """Initialize the environment and set random seeds"""
        if self.args.env_name == "LunarLander-v2":
            self.env = gym.make(self.args.env_name, continuous=True)
        else:
            self.env = gym.make(self.args.env_name)

        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            self.env.action_space.seed(self.args.seed)

    def setup_agent(self):
        """Initialize the SAC agent"""
        critic_loss_fn = select_loss_function(self.args.loss_type)

        self.agent = SACAgent(
            self.env.observation_space,
            self.env.action_space,
            critic_loss_fn,
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
        )

    def get_run_name(self):
        """Generate a descriptive run name based on hyperparameters"""
        components = [
            f"SAC",
            f"env_{self.args.env_name}",
            f"lr_{self.args.lr}",
            f"seed_{self.args.seed}" if self.args.seed is not None else "seed_none",
        ]

        # Add boolean flags if they're True
        if self.args.use_per:
            components.append(f"PER_a{self.args.per_alpha}_b{self.args.per_beta}")
        if self.args.use_ere:
            components.append(f"ERE_eta{self.args.ere_eta0}")

        # Add loss type if it's not the default
        if self.args.loss_type != "mse":
            components.append(f"loss_{self.args.loss_type}")

        # Add timestamp at the end
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        components.append(timestamp)

        return "-".join(components)

    def setup_logging(self):
        """Setup logging directories and files"""
        self.run_name = self.get_run_name()
        self.output_dir = Path(self.args.output_dir) / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(vars(self.args), f, indent=4)

        # Initialize metrics
        self.rewards = []
        self.lengths = []
        self.losses = []
        self.timestep = 0

        # Create a summary file with hyperparameters
        self.create_summary_file()

        np.savez(
            self.output_dir / "rewards.npz",
            rewards=np.array([]),
            episode_lengths=np.array([]),
            moving_avg_100=np.array([]),
            config=json.dumps(vars(self.args)),
        )

    def create_summary_file(self):
        """Create a human-readable summary of the experiment"""
        summary = [
            "Experiment Summary",
            "=" * 50,
            "\nEnvironment Configuration:",
            f"Environment: {self.args.env_name}",
            f"Max Episodes: {self.args.max_episodes}",
            f"Max Timesteps per Episode: {self.args.max_timesteps}",
            "\nTraining Configuration:",
            f"Learning Rate: {self.args.lr}",
            f"Loss Type: {self.args.loss_type}",
            f"Target Network Update Frequency: {self.args.update_every}",
            f"Random Seed: {self.args.seed}",
            "\nPER Configuration:",
            f"Enabled: {self.args.use_per}",
            f"Alpha: {self.args.per_alpha}",
            f"Beta: {self.args.per_beta}",
            "\nERE Configuration:",
            f"Enabled: {self.args.use_ere}",
            f"Eta0: {self.args.ere_eta0}",
            f"Min Size: {self.args.ere_min_size}",
            "\nLogging Configuration:",
            f"Output Directory: {self.args.output_dir}",
            f"Save Interval: {self.args.save_interval}",
            f"Log Interval: {self.args.log_interval}",
            "\nRun Name:",
            self.run_name,
        ]

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
                print(
                    f"Episode {episode} \t avg length: {avg_length} \t reward: {avg_reward}"
                )

        # Final save
        self.save_checkpoint("final")
        self.plot_metrics()


def parse_args():
    parser = argparse.ArgumentParser(description="SAC trainer")
    parser.add_argument(
        "--env_name", type=str, default="Pendulum-v1", help="Environment name"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--max_episodes", type=int, default=2000, help="Maximum number of episodes"
    )
    parser.add_argument(
        "--max_timesteps", type=int, default=2000, help="Maximum timesteps per episode"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        help="Loss function type (mse/huber/mae/mse_weighted)",
    )
    parser.add_argument(
        "--update_every", type=float, default=1, help="Target network update frequency"
    )

    # PER and ERE parameters
    parser.add_argument(
        "--use_per",
        action="store_true",
        help="Use Prioritized Experience Replay",
        default=False,
    )
    parser.add_argument(
        "--use_ere",
        action="store_true",
        help="Use Emphasizing Recent Experience",
        default=False,
    )
    parser.add_argument(
        "--per_alpha", type=float, default=0.6, help="PER alpha parameter"
    )
    parser.add_argument(
        "--per_beta", type=float, default=0.4, help="PER beta parameter"
    )
    parser.add_argument(
        "--ere_eta0", type=float, default=0.996, help="ERE initial eta parameter"
    )
    parser.add_argument(
        "--ere_min_size", type=int, default=2500, help="ERE minimum buffer size"
    )

    # Logging parameters
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="Output directory"
    )
    parser.add_argument(
        "--save_interval", type=int, default=500, help="Save checkpoint interval"
    )
    parser.add_argument("--log_interval", type=int, default=20, help="Logging interval")

    parser.add_argument(
        "--noise_type",
        type=str,
        default="normal",
        choices=["normal", "ornstein", "colored", "pink"],
        help="Type of action noise (normal/ornstein/colored/pink)",
    )
    # General noise parameters
    parser.add_argument(
        "--noise_sigma", type=float, default=0.1, help="Noise sigma/scale parameter"
    )
    # Ornstein-Uhlenbeck specific
    parser.add_argument(
        "--noise_theta", type=float, default=0.15, help="OU noise theta parameter"
    )
    parser.add_argument(
        "--noise_dt", type=float, default=1e-2, help="OU noise dt parameter"
    )
    # Colored noise specific
    parser.add_argument(
        "--noise_beta", type=float, default=1.0, help="Colored noise beta parameter"
    )
    parser.add_argument(
        "--noise_seq_len",
        type=int,
        default=1000,
        help="Sequence length for colored/pink noise",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
