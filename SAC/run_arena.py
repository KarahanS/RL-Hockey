#!/usr/bin/env python
"""
Wrapper Script to Run the Agents Evaluation Runner Directly

This script defines the required parameters (such as paths to the config and checkpoint files,
evaluation episodes, environment mode, opponent type, and whether to render) inside the file.
It then simulates the command-line arguments (as expected by agent_arena.py)
by setting sys.argv accordingly and calls the main() function from agent_arena.py.

Differences from the original:
  - Adds a '--agent1_type' argument (which can be 'sac' or 'td3').
  - The rest is the same logic, just pointing to your updated arena script
    which can handle agent1 as either SAC or TD3.
"""

import sys
import os

# Add the 'src' directory to the Python path so that modules like hockey_env can be found.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# If your updated evaluation script is called agent_arena.py, import that.
# Otherwise, if it's still named sac_arena.py, update this import as needed.
import sac_arena  # or "import sac_arena" if that's your updated file name.


def main():
    # Define the variables you want to use.

    # Agent1 type: "sac", "td3", or "dqn"
    agent1_type = "sac"  # Switch to "td3" if you want agent1 to be a TD3 agent.
    
    # Paths for agent1's config and checkpoint.
    #agent1_config = "td3_models/td3_vs_selfplay_v2/config.json"
    #agent1_checkpoint = "td3_models/td3_vs_selfplay_v2/TD3_Hockey_self_play_seed_700_final.pth"

    agent1_config = "final_train/config.json"
    agent1_checkpoint = "final_train/PINK_01_STRONG.pth"
    
    # For agent2, you can use either SAC or TD3 or a built-in type.
    agent2_config = "final_train/config.json"
    agent2_checkpoint = "final_train/PINK_01_WEAK_SELFPLAY_MODE1_6000_TH4_MIRROR.pth"

    # For a TD3 example: "pth/td3_self_play/TD3_Hockey_self_play_seed_700_final"

    # Set opponent_type to: "sac", "td3", "weak", "strong", "basicdefense", "basicattack", or "none"
    opponent_type = "sac"
    
        
    print("Running evaluation with:")
    print("Agent1:", agent1_type, agent1_config, agent1_checkpoint)
    print("Opponent type:", opponent_type)
    if agent1_type == "sac" or agent1_type == "td3":
        print("Agent2 (SAC):", agent2_config, agent2_checkpoint)
    else:
        print("Agent2 (Built-in):", agent2_config, agent2_checkpoint)

    eval_episodes = 10000  # Number of evaluation episodes
    env_mode = "NORMAL"   # Could be TRAIN_SHOOTING, TRAIN_DEFENSE, etc.
    render = False

    # Now we build sys.argv to pass to the updated agent_arena.py or sac_arena.py
    sys.argv = [
        "agent_arena.py",  # dummy script name
        "--agent1_type", agent1_type,
        "--agent1_config", agent1_config,
        "--agent1_checkpoint", agent1_checkpoint,
        "--agent2_config", agent2_config,
        "--agent2_checkpoint", agent2_checkpoint,
        "--opponent_type", opponent_type,
        "--eval_episodes", str(eval_episodes),
        "--env_mode", env_mode
    ]

    if render:
        sys.argv.append("--render")

    # Call the main function from agent_arena.py (or sac_arena if that's your updated script).
    results = sac_arena.main()  # results is typically a triple: (agent1win, agent2win, draw)
    # or you can do:
    # (agent1win, agent2win, tie) = agent_arena.main()

    # If you wish, you can print the results here:
    # print("Evaluation done. Agent1Wins:", agent1win, "Agent2Wins:", agent2win, "Ties:", tie)


if __name__ == "__main__":
    main()
