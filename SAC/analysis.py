import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse

def load_rewards(npz_file):
    """Load rewards data from npz file"""
    data = np.load(npz_file, allow_pickle=True)
    rewards = data['rewards']
    moving_avg = data['moving_avg_100']
    config = json.loads(str(data['config']))  # Convert from numpy string to dict
    return rewards, moving_avg, config

def plot_multiple_rewards(result_dirs, save_path=None, window_size=100):
    """Plot multiple reward curves on the same plot
    
    Args:
        result_dirs (list): List of directories containing reward.npz files
        save_path (str, optional): Where to save the plot. If None, displays it
        window_size (int): Window size for moving average if not pre-computed
    """
    plt.figure(figsize=(12, 8))
    
    for result_dir in result_dirs:
        path = Path(result_dir)
        rewards_file = path / "rewards.npz"
        
        if not rewards_file.exists():
            print(f"Warning: No rewards.npz found in {result_dir}")
            continue
            
        rewards, moving_avg, config = load_rewards(rewards_file)
        
        # Create label from config
        label = f"lr={config['lr']}"
        if config['use_per']:
            label += f"_PER"
        if config['use_ere']:
            label += f"_ERE"
        label += f"_loss={config['loss_type']}"
        
        # Plot moving average
        if len(moving_avg) > 0:
            plt.plot(moving_avg, label=label)
        else:
            # If moving average wasn't pre-computed, compute it now
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(moving_avg, label=label)
    
    plt.title(f"Training Rewards Comparison (Moving Avg {window_size} episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot multiple reward curves")
    parser.add_argument('--dirs', nargs='+', help='List of result directories to plot', required=True)
    parser.add_argument('--save_path', type=str, help='Path to save the plot (optional)')
    parser.add_argument('--window', type=int, default=100, help='Window size for moving average')
    
    args = parser.parse_args()
    plot_multiple_rewards(args.dirs, args.save_path, args.window)

if __name__ == "__main__":
    main()