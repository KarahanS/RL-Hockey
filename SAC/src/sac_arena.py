#!/usr/bin/env python
"""
Evaluation Script for Two Agents Playing Against Each Other

This script loads a SACAgent instance for agent1 from its config and checkpoint.
For agent2, it is possible to choose between:
  - A trained SACAgent (by setting --opponent_type sac),
  - A trained TD3 agent (by setting --opponent_type td3),
  - A weak BasicOpponent (by setting --opponent_type weak),
  - A strong BasicOpponent (by setting --opponent_type strong),
  - or a dummy opponent (by setting --opponent_type none).

It then evaluates the two agents by letting them play against each other.
It supports loading the agent’s configuration from the checkpoint (if available)
or from a separate config file, as well as specifying the number of evaluation
episodes and an option to render the gameplay.
"""

# add the project root to the path so that TD3 is available as a package
import sys

sys.path.append("../")  # Adjust this if needed

# Import TD3 from its package (assuming TD3/src is part of the TD3 package)
from TD3.src.td3 import TD3

import argparse
import torch
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from hockey_env import HockeyEnv, Mode, BasicOpponent
from sac import SACAgent
import time


def load_sac_agent(config_path, checkpoint_path, env):
    """
    Loads a SACAgent from a checkpoint.

    First, it tries to extract the 'config' from the checkpoint.
    If that key is not present, it loads the configuration from the provided config file.
    Then, it converts the hidden layer size strings into lists of integers,
    converts the learn_alpha parameter to a boolean (if needed),
    creates a SACAgent instance (using the environment’s observation and action spaces),
    and restores its state from the checkpoint.
    """
    # Load the checkpoint (forcing CPU map location for compatibility)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Try to get the config from the checkpoint; if not found, load from the config file.
    if "config" in checkpoint:
        config = checkpoint["config"]
        print("Loaded configuration from checkpoint.")
    else:
        with open(config_path, "r") as f:
            config = json.load(f)
        print("Loaded configuration from config file.")

    # If checkpoint is a tuple of 3 elements, convert it into a dictionary.
    if isinstance(checkpoint, tuple):
        tpl = checkpoint
        checkpoint = {}
        checkpoint["actor_state_dict"] = tpl[0]
        checkpoint["critic1_state_dict"] = tpl[1]
        checkpoint["critic2_state_dict"] = tpl[2]

    # Convert comma-separated strings into lists of integers for the hidden layer sizes.
    if type(config["hidden_sizes_actor"]) == str:
        config["hidden_sizes_actor"] = list(
            map(int, config["hidden_sizes_actor"].split(","))
        )
    if type(config["hidden_sizes_critic"]) == str:
        config["hidden_sizes_critic"] = list(
            map(int, config["hidden_sizes_critic"].split(","))
        )

    # Convert learn_alpha to boolean if it is provided as a string.
    learn_alpha = config["learn_alpha"]
    if isinstance(learn_alpha, str):
        learn_alpha = learn_alpha.lower() == "true"

    # Create the SACAgent instance using the environment's spaces and the configuration.
    agent = SACAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        discount=config.get("discount", 0.99),
        buffer_size=config.get("buffer_size", int(1e6)),
        learning_rate_actor=config.get("learning_rate_actor", 3e-4),
        learning_rate_critic=config.get("learning_rate_critic", 3e-4),
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
                "type": "normal",
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

    # Restore the agent's full state (including networks and optimizers) from the checkpoint.
    agent.restore_full_state(checkpoint)
    return agent


def load_td3_agent(config_path, checkpoint_prefix, env):
    """
    Loads a TD3 agent from a JSON config and checkpoint prefix.

    The TD3 checkpoint is assumed to be saved as multiple files with the prefix.
    For example, if checkpoint_prefix is "path/to/checkpoint", then the following
    files should exist:
       - path/to/checkpoint_actor.pth
       - path/to/checkpoint_critic.pth
       - path/to/checkpoint_actor_optimizer.pth
       - path/to/checkpoint_critic_optimizer.pth
       - etc. (if RND is used)
    """
    # Load the configuration from the provided config file.
    with open(config_path, "r") as f:
        config = json.load(f)
    print("Loaded TD3 configuration from config file.")

    # Determine dimensions.
    # Note: In this environment, the full action space is for two agents.
    # Here, we assume TD3 controls the second agent, so its action dimension is half.
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2
    max_action = env.action_space.high[0]  # Assumes symmetric action space.

    # Create the TD3 agent.
    agent = TD3(state_dim, action_dim, max_action, training_config=config)
    # Load the checkpoint state.
    import copy

    agent.critic.load_state_dict(
        torch.load(
            checkpoint_prefix + "_critic.pth",
            weights_only=True,
            map_location=torch.device("cpu"),
        )
    )
    agent.critic_optimizer.load_state_dict(
        torch.load(
            checkpoint_prefix + "_critic_optimizer.pth",
            weights_only=True,
            map_location=torch.device("cpu"),
        )
    )
    agent.critic_target = copy.deepcopy(agent.critic)

    agent.actor.load_state_dict(
        torch.load(
            checkpoint_prefix + "_actor.pth",
            weights_only=True,
            map_location=torch.device("cpu"),
        )
    )
    agent.actor_optimizer.load_state_dict(
        torch.load(
            checkpoint_prefix + "_actor_optimizer.pth",
            weights_only=True,
            map_location=torch.device("cpu"),
        )
    )
    agent.actor_target = copy.deepcopy(agent.actor)

    if agent.use_rnd:
        agent.rnd.target_network.load_state_dict(
            torch.load(
                checkpoint_prefix + "_rnd_target.pth",
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )
        agent.rnd.predictor_network.load_state_dict(
            torch.load(
                checkpoint_prefix + "_rnd_predictor.pth",
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )
        agent.rnd.optimizer.load_state_dict(
            torch.load(
                checkpoint_prefix + "_rnd_optimizer.pth",
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )
    return agent


def evaluate_agents(agent1, agent2, env, eval_episodes=100, render=False):
    """
    Evaluates two agents by letting them play for a specified number of episodes.
    """
    results = {"agent1_win": 0, "agent2_win": 0, "draw": 0}
    for episode in range(eval_episodes):
        obs, _ = env.reset()
        opponent_obs = env.obs_agent_two() if hasattr(env, "obs_agent_two") else obs
        done = False
        while not done:
            # For agent1 (always SAC), we pass eval_mode=True.
            action1 = agent1.act(obs, eval_mode=True)

            # For agent2, we check if it's a TD3 agent (or anything else) and call appropriately.
            if isinstance(agent2, TD3):
                # For TD3, disable noise during evaluation.
                action2 = agent2.act(opponent_obs, add_noise=False)
            else:
                try:
                    action2 = agent2.act(opponent_obs, eval_mode=True)
                except TypeError:
                    action2 = agent2.act(opponent_obs)

            # The full action is the concatenation of agent1 and agent2 actions.
            full_action = np.hstack([action1, action2])
            obs, reward, done, _, info = env.step(full_action)
            opponent_obs = env.obs_agent_two() if hasattr(env, "obs_agent_two") else obs
            if render:
                env.render("human")
                time.sleep(1.0 / 50)
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
        description="Evaluate SAC agent vs. opponent in the Hockey Environment"
    )
    parser.add_argument(
        "--agent1_config",
        type=str,
        required=True,
        help="Path to the JSON config for agent1 (SAC)",
    )
    parser.add_argument(
        "--agent1_checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint for agent1 (SAC)",
    )
    # These are used if the opponent is set to 'sac' (SAC) or 'td3'
    parser.add_argument(
        "--agent2_config",
        type=str,
        default="",
        help="Path to the JSON config for agent2 (used if opponent_type is 'sac' or 'td3')",
    )
    parser.add_argument(
        "--agent2_checkpoint",
        type=str,
        default="",
        help="Path to the checkpoint for agent2 (used if opponent_type is 'sac' or 'td3')",
    )
    parser.add_argument(
        "--opponent_type",
        type=str,
        default="sac",
        help="Type of opponent: sac, td3, weak, strong, none",
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=100, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--env_mode",
        type=str,
        default="NORMAL",
        help="Hockey environment mode (e.g., NORMAL)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render the environment during evaluation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = Mode[args.env_mode.upper()]
    env = HockeyEnv(mode=mode)

    # Load agent1 as a SACAgent.
    agent1 = load_sac_agent(args.agent1_config, args.agent1_checkpoint, env)

    # For agent2, determine based on opponent_type.
    opponent_type = args.opponent_type.lower()
    if opponent_type == "sac":
        if not args.agent2_config or not args.agent2_checkpoint:
            raise ValueError(
                "For sac opponent, agent2_config and agent2_checkpoint must be provided."
            )
        agent2 = load_sac_agent(args.agent2_config, args.agent2_checkpoint, env)
    elif opponent_type == "td3":
        if not args.agent2_config or not args.agent2_checkpoint:
            raise ValueError(
                "For TD3 opponent, agent2_config and agent2_checkpoint must be provided."
            )
        agent2 = load_td3_agent(args.agent2_config, args.agent2_checkpoint, env)
    elif opponent_type in ["weak", "strong"]:
        is_weak = opponent_type == "weak"
        agent2 = BasicOpponent(weak=is_weak, keep_mode=env.keep_mode)
    elif opponent_type == "none":
        # Create a dummy opponent that always returns a zero-action.
        class DummyOpponent:
            def act(self, obs, **kwargs):
                # Assume half the size of the full action space.
                return np.zeros(env.action_space.shape[0] // 2, dtype=np.float32)

        agent2 = DummyOpponent()
    else:
        raise ValueError(f"Unknown opponent type: {args.opponent_type}")

    # Set networks to evaluation mode if applicable.
    agent1.actor.eval()
    agent1.critic1.eval()
    agent1.critic2.eval()
    if opponent_type in ["sac", "td3"]:
        try:
            agent2.actor.eval()
            agent2.critic1.eval()
            agent2.critic2.eval()
        except AttributeError:
            pass

    results = evaluate_agents(
        agent1, agent2, env, eval_episodes=args.eval_episodes, render=args.render
    )

    print("Evaluation Results:")
    print(f"Agent1 Wins: {results['agent1_win']} / {args.eval_episodes}")
    print(f"Agent2 Wins: {results['agent2_win']} / {args.eval_episodes}")
    print(f"Draws: {results['draw']} / {args.eval_episodes}")

    labels = ["Agent1", "Agent2", "Draw"]
    values = [results["agent1_win"], results["agent2_win"], results["draw"]]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=["blue", "green", "gray"])
    plt.title("Evaluation Results")
    plt.ylabel("Number of Wins")
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )
    plot_filename = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    # save as pdf
    plt.savefig(plot_filename)
    plt.show()
    # print(f"Saved evaluation plot as {plot_filename}")


if __name__ == "__main__":
    main()
