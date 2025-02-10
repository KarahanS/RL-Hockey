#!/usr/bin/env python
"""
Generic Evaluation Script for Opponent Play

This script loads two agents from their respective config JSON files and checkpoint files.
They may be implemented with different algorithms (e.g., SAC, TD3, DDQN).
It then evaluates them by letting them play against each other.
"""

import argparse
import torch
import json
import numpy as np
import os
import importlib  # For dynamic agent loading
from hockey_env import HockeyEnv, Mode
import matplotlib.pyplot as plt
from datetime import datetime


def load_agent_from_checkpoint(checkpoint_path, env):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    config = checkpoint.get("config", {})  # Default to empty dict if missing

    # If agent_class is missing, assume it's SAC
    agent_class_str = config.get("agent_class", "sac:SACAgent")
    print(f"Loaded Agent 1 class: {agent_class_str}")  # Debug

    module_name, class_name = agent_class_str.split(":")
    agent_class = getattr(importlib.import_module(module_name), class_name)

    agent = agent_class(
        observation_space=env.observation_space,
        action_space=env.action_space,
        **config
    )
    agent.load_state_dict(checkpoint["model_state_dict"])
    return agent


# Generic helper to load an agent from config and checkpoint.
def load_agent_from_config(config_path, checkpoint_path, env):
    with open(config_path, "r") as f:
        config = json.load(f)
    agent_class_str = config.get("agent_class")
    if agent_class_str is None:
        raise ValueError("Config must include an 'agent_class' key.")
    module_name, class_name = agent_class_str.split(":")
    agent_class = getattr(importlib.import_module(module_name), class_name)
    # Instantiate the agent using environment spaces and config parameters.
    agent = agent_class(
        observation_space=env.observation_space,
        action_space=env.action_space,
        loss_fn=None,  # Assume the loss function is set internally or via config.
        **config  # Assumes the config keys match the agent’s __init__ signature.
    )
    checkpoint = torch.load(checkpoint_path)
    agent.load_state_dict(checkpoint)
    return agent, config

def evaluate_agents(agent1, agent2, env, eval_episodes=100):
    results = {"agent1_win": 0, "agent2_win": 0, "draw": 0}
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # Each agent selects an action.
            action1 = agent1.act(obs, eval_mode=True)
            action2 = agent2.act(obs, eval_mode=True)
            full_action = np.hstack([action1, action2])
            obs, reward, done, info = env.step(full_action)
        # Decide winner based on environment info (expects info["winner"]: 1 for agent1, -1 for agent2, 0 for draw)
        if "winner" in info:
            if info["winner"] == 1:
                results["agent1_win"] += 1
            elif info["winner"] == -1:
                results["agent2_win"] += 1
            else:
                results["draw"] += 1
    return results

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate two agents (of potentially different algorithms) against each other in the Hockey Environment"
    )
    parser.add_argument("--agent1_config", type=str, required=True, help="Path to the JSON config for agent1")
    parser.add_argument("--agent1_checkpoint", type=str, required=True, help="Path to the checkpoint for agent1")
    parser.add_argument("--agent2_config", type=str, required=True, help="Path to the JSON config for agent2")
    parser.add_argument("--agent2_checkpoint", type=str, required=True, help="Path to the checkpoint for agent2")
    parser.add_argument("--eval_episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--env_mode", type=str, default="NORMAL", help="Hockey environment mode (e.g., NORMAL)")
    return parser.parse_args()

def main():
    args = parse_args()
    mode = Mode[args.env_mode.upper()]
    env = HockeyEnv(mode=mode, render=True)
    
    # Load both agents using the generic loader.
    agent1, config1 = load_agent_from_config(args.agent1_config, args.agent1_checkpoint, env)
    agent2, config2 = load_agent_from_config(args.agent2_config, args.agent2_checkpoint, env)
    
    # Set agents to evaluation mode if applicable.
    agent1.actor.eval()
    agent1.critic.eval()
    agent2.actor.eval()
    agent2.critic.eval()
    
    results = evaluate_agents(agent1, agent2, env, eval_episodes=args.eval_episodes)
    print("Evaluation Results:")
    print(f"Agent1 Wins: {results['agent1_win']} / {args.eval_episodes}")
    print(f"Agent2 Wins: {results['agent2_win']} / {args.eval_episodes}")
    print(f"Draws: {results['draw']} / {args.eval_episodes}")
    
    # Plotting the results.
    labels = ['Agent1', 'Agent2', 'Draw']
    values = [results['agent1_win'], results['agent2_win'], results['draw']]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=['blue', 'green', 'gray'])
    plt.title("Evaluation Results")
    plt.ylabel("Number of Wins")
    plot_filename = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_filename)
    plt.show()
    print(f"Saved evaluation plot as {plot_filename}")

if __name__ == "__main__":
    main()
