#!/usr/bin/env python
"""
Evaluation Script for Two Agents Playing Against Each Other

Now supports agent1 being either SAC or TD3 (controlled by --agent1_type).
Agent2 can be: 
  - 'sac' (a SACAgent),
  - 'td3' (a TD3 agent),
  - 'weak' or 'strong' (BasicOpponent),
  - 'basicdefense' or 'basicattack' (special opponents),
  - or 'none' (dummy that does nothing).

We load agent1 from the provided config+checkpoint, 
and agent2 from config+checkpoint or from built-in opponents as specified.
Then run them head-to-head for a certain number of episodes.
"""

import sys
sys.path.append("../")  # Adjust if needed so that TD3, etc. are importable.


from DDQN.DDQN import DoubleDuelingDQNAgent
from DDQN.action_space import CustomActionSpace
from TD3.src.td3 import TD3  # your TD3 implementation
import argparse
import torch
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import time

from hockey_env import HockeyEnv, Mode, BasicOpponent, BasicAttackOpponent, BasicDefenseOpponent
from sac import SACAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sac_agent(config_path, checkpoint_path, env):
    """
    Loads a SACAgent from a checkpoint (same as your original function).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Try config from checkpoint
    if "config" in checkpoint:
        config = checkpoint["config"]
        print("Loaded SAC config from checkpoint.")
    else:
        raise ValueError("No config found in checkpoint.")

    learn_alpha = config.get("learn_alpha", True)
    if isinstance(learn_alpha, str):
        learn_alpha = (learn_alpha.lower() == "true")

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
        control_half=True,
    )
    agent.restore_full_state(checkpoint)
    return agent

def load_td3_agent(config_path, checkpoint_prefix, env):
    """
    Loads a TD3 agent from a JSON config and checkpoint prefix (same as original).
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    print("Loaded TD3 configuration from:", config_path)

    # If agent1 is also controlling only half the action space, keep consistent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2
    max_action = env.action_space.high[0]

    agent = TD3(state_dim, action_dim, max_action, training_config=config)
    import copy

    agent.critic.load_state_dict(
        torch.load(checkpoint_prefix + "_critic.pth", map_location=device)
    )
    agent.critic_optimizer.load_state_dict(
        torch.load(checkpoint_prefix + "_critic_optimizer.pth", map_location=device)
    )
    agent.critic_target = copy.deepcopy(agent.critic)

    agent.actor.load_state_dict(
        torch.load(checkpoint_prefix + "_actor.pth", map_location=device)
    )
    agent.actor_optimizer.load_state_dict(
        torch.load(checkpoint_prefix + "_actor_optimizer.pth", map_location=device)
    )
    agent.actor_target = copy.deepcopy(agent.actor)

    if agent.use_rnd:
        agent.rnd.target_network.load_state_dict(
            torch.load(checkpoint_prefix + "_rnd_target.pth", map_location=device)
        )
        agent.rnd.predictor_network.load_state_dict(
            torch.load(checkpoint_prefix + "_rnd_predictor.pth", map_location=device)
        )
        agent.rnd.optimizer.load_state_dict(
            torch.load(checkpoint_prefix + "_rnd_optimizer.pth", map_location=device)
        )
    return agent

def load_dqn_agent(config_path, checkpoint_path, env):
    agent = DoubleDuelingDQNAgent(
        env.observation_space,
        CustomActionSpace(),
        hidden_sizes=[512],
        hidden_sizes_A=[512, 512],
        hidden_sizes_V=[512, 512],
        use_torch=True
    )
    agent.load_state(checkpoint_path)
    return agent



    act_a1_discr = agent.act_torch(np2gpu(ob_a1), explore=True)
    act_a1 = CustomActionSpace.discrete_to_continuous(act_a1_discr)
    
def load_agent(agent_type, config_path, checkpoint_path, env):
    """
    Helper that unifies either SAC or TD3 for agent1.
    """
    if agent_type.lower() == "sac":
        return load_sac_agent(config_path, checkpoint_path, env)
    elif agent_type.lower() == "td3":
        return load_td3_agent(config_path, checkpoint_path, env)
    elif agent_type.lower() == "dqn":
        return load_dqn_agent(config_path, checkpoint_path, env)
    else:
        raise ValueError(f"Invalid agent1_type: {agent_type}, must be sac or td3.")


def evaluate_agents(agent1, agent2, env, eval_episodes=100, render=False):
    """
    Evaluates two agents by letting them play for a specified number of episodes.
    If agent1 is SAC or TD3, we handle them similarly, disabling exploration.
    Same for agent2.
    """
    results = {"agent1_win": 0, "agent2_win": 0, "draw": 0}

    def np2gpu(data: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(data).float().to(device)
    
    for episode in range(eval_episodes):
        obs, _ = env.reset()
        opp_obs = env.obs_agent_two() if hasattr(env, "obs_agent_two") else obs
        done = False

        while not done:
            # ----- Agent1 action -----
            if isinstance(agent1, SACAgent):
                # Evaluate with no noise
                action1 = agent1.act(obs, eval_mode=True)
            elif isinstance(agent1, TD3):
                # Evaluate with no noise
                action1 = agent1.act(obs, add_noise=False)
            elif isinstance(agent1, DoubleDuelingDQNAgent):
                action1_discrete  = agent1.act_torch(np2gpu(obs))  # int
                action1 = env.discrete_to_continous_action(action1_discrete)  # numpy array
            else:
                # Possibly a custom or dummy agent
                try:
                    action1 = agent1.act(obs, eval_mode=True)
                except TypeError:
                    action1 = agent1.act(obs)

            # ----- Agent2 action -----
            if isinstance(agent2, SACAgent):
                action2 = agent2.act(opp_obs, eval_mode=True)
            elif isinstance(agent2, TD3):
                action2 = agent2.act(opp_obs, add_noise=False)
            elif isinstance(agent2, DoubleDuelingDQNAgent):
                action2_discrete  = agent2.act_torch(np2gpu(obs))  # int
                action2 = env.discrete_to_continous_action(action2_discrete)  # numpy array
            else:
                try:
                    action2 = agent2.act(opp_obs, eval_mode=True)
                except TypeError:
                    action2 = agent2.act(opp_obs)

            full_action = np.hstack([action1, action2])
            obs, reward, done, _, info = env.step(full_action)
            opp_obs = env.obs_agent_two() if hasattr(env, "obs_agent_two") else obs

            if render:
                env.render("human")
                time.sleep(1.0 / 60)

        # final outcome
        if "winner" in info:
            if info["winner"] == 1:
                results["agent1_win"] += 1
            elif info["winner"] == -1:
                results["agent2_win"] += 1
            else:
                results["draw"] += 1

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate two Hockey agents (SAC or TD3).")
    parser.add_argument("--agent1_type", type=str, default="sac",
                        help="Type for agent1: sac or td3.")
    parser.add_argument("--agent1_config", type=str, required=True,
                        help="Path to the JSON config for agent1.")
    parser.add_argument("--agent1_checkpoint", type=str, required=True,
                        help="Path to the checkpoint (prefix if td3) for agent1.")
    parser.add_argument("--agent2_config", type=str, default="",
                        help="Path to the JSON config for agent2 if sac or td3.")
    parser.add_argument("--agent2_checkpoint", type=str, default="",
                        help="Checkpoint (prefix if td3) for agent2 if sac or td3.")
    parser.add_argument("--opponent_type", type=str, default="sac",
                        help="Type of opponent: sac, td3, weak, strong, none, basicdefense, basicattack")
    parser.add_argument("--eval_episodes", type=int, default=100,
                        help="Number of evaluation episodes.")
    parser.add_argument("--env_mode", type=str, default="NORMAL",
                        help="Hockey environment mode (NORMAL, TRAIN_SHOOTING, etc).")
    parser.add_argument("--render", action="store_true", default=False,
                        help="Render gameplay if set.")
    return parser.parse_args()


def main():
    args = parse_args()
    mode = Mode[args.env_mode.upper()]
    env = HockeyEnv(mode=mode)

    # ------ Load agent1 (SAC or TD3) ------
    print(f"Loading agent1 as {args.agent1_type.upper()} ...")
    agent1 = load_agent(
        agent_type=args.agent1_type,
        config_path=args.agent1_config,
        checkpoint_path=args.agent1_checkpoint,
        env=env
    )

    # ------ Load agent2 or create built-in. ------
    opp_type = args.opponent_type.lower()
    if opp_type in ["sac", "td3", "dqn"]:
        if not args.agent2_config or not args.agent2_checkpoint:
            raise ValueError(f"For {opp_type} opponent, must provide --agent2_config and --agent2_checkpoint.")
        if opp_type == "sac":
            agent2 = load_sac_agent(args.agent2_config, args.agent2_checkpoint, env)
        elif opp_type == "td3":
            agent2 = load_td3_agent(args.agent2_config, args.agent2_checkpoint, env)
        else:
            agent2 = load_dqn_agent(args.agent2_config, args.agent2_checkpoint, env)
    elif opp_type == "weak":
        agent2 = BasicOpponent(weak=True)
    elif opp_type == "strong":
        agent2 = BasicOpponent(weak=False)
    elif opp_type == "none":
        class DummyOpponent:
            def act(self, observation, **kwargs):
                return np.zeros(env.action_space.shape[0] // 2, dtype=np.float32)
        agent2 = DummyOpponent()
    elif opp_type == "basicdefense":
        agent2 = BasicDefenseOpponent()
    elif opp_type == "basicattack":
        agent2 = BasicAttackOpponent()
    else:
        raise ValueError(f"Unknown opponent type: {opp_type}")

    # Put agent1 and agent2 in evaluation mode if they have networks
    if isinstance(agent1, SACAgent):
        agent1.actor.eval()
        agent1.critic1.eval()
        agent1.critic2.eval()
    elif isinstance(agent1, TD3):
        # Typically no separate "eval" method, but you can do so if you want
        pass

    if isinstance(agent2, SACAgent):
        agent2.actor.eval()
        agent2.critic1.eval()
        agent2.critic2.eval()
    elif isinstance(agent2, TD3):
        # same note as above
        pass

    # ------- Evaluate! -------
    results = evaluate_agents(agent1, agent2, env, eval_episodes=args.eval_episodes, render=args.render)

    print("Evaluation Results:")
    print(f"  Agent1 Wins: {results['agent1_win']} / {args.eval_episodes}")
    print(f"  Agent2 Wins: {results['agent2_win']} / {args.eval_episodes}")
    print(f"  Draws:       {results['draw']}      / {args.eval_episodes}")

    # Quick bar plot
    labels = ["Agent1", "Agent2", "Draw"]
    values = [results["agent1_win"], results["agent2_win"], results["draw"]]
    plt.figure(figsize=(6,4))
    bars = plt.bar(labels, values, color=["blue","green","gray"])
    plt.title("Evaluation Results")
    plt.ylabel("Number of Wins")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2.0, height, f"{int(height)}",
                 ha="center", va="bottom")
    # Optional saving
    # plot_filename = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    # plt.savefig(plot_filename)
    # plt.show()

    return values


if __name__ == "__main__":
    main()
