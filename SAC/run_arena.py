#!/usr/bin/env python
"""
Wrapper Script to Run the SAC Agents Evaluation Runner Directly

This script defines the required parameters (such as paths to the config and checkpoint files,
evaluation episodes, environment mode, opponent type, and whether to render) inside the file.
It then simulates the command-line arguments (as expected by sac_arena.py)
by setting sys.argv accordingly and calls the main() function from sac_arena.py.
"""

import sys
import os

# Add the 'src' directory to the Python path so that modules like hockey_env can be found.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import sac_arena  # Ensure sac_arena.py is in your PYTHONPATH or in the same directory.


def main():
    # Define the variables you want to use.
    # For agent1 (SAC agent) you provide its config and checkpoint.
    agent1_config = "pth/alpha0/config.json"  # For example, if config is embedded in the checkpoint.
    agent1_checkpoint = "pth/alpha0/alpha0_9000.pth"

    # For agent2, you can use either SAC or TD3.
    # If you choose TD3, make sure the config and checkpoint paths point to your TD3 files.
    agent2_config = "pth/alpha0/config.json"  # Update this if using TD3.
    agent2_checkpoint = "pth/alpha0/alpha0_8500.pth"  # Update this if using TD3.
    # "pth/td3_self_play/TD3_Hockey_self_play_seed_700_final.pth"

    # Set opponent_type to one of: "sac", "td3", "weak", "strong", or "none"
    opponent_type = "sac"  # Change to "td3" to use a TD3 opponent.

    eval_episodes = 10000  # Number of evaluation episodes
    env_mode = "NORMAL"  # Environment mode (e.g., NORMAL, TRAIN_SHOOTING, etc.)
    render = False  # Set to True if you wish to see the gameplay rendered

    # Simulate the command-line arguments expected by sac_arena.py.
    # Note: The "--opponent_type" argument now supports "td3" as well.
    sys.argv = [
        "sac_arena.py",  # Dummy script name
        "--agent1_config",
        agent1_config,
        "--agent1_checkpoint",
        agent1_checkpoint,
        "--agent2_config",
        agent2_config,
        "--agent2_checkpoint",
        agent2_checkpoint,
        "--opponent_type",
        opponent_type,
        "--eval_episodes",
        str(eval_episodes),
        "--env_mode",
        env_mode,
    ]

    # Append the render flag if rendering is desired.
    if render:
        sys.argv.append("--render")

    # Call the main function from sac_arena.py.
    (agent1win, agent2win, tie) = sac_arena.main()


if __name__ == "__main__":
    main()
