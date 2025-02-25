#!/usr/bin/env python
"""
Evaluate multiple agents (SAC/TD3) + built-in opponents in a round-robin fashion
and track their ratings using TrueSkill (similar to an Elo system).

It scans the specified `--model_folder` *recursively* for *.pth files, 
and tries to load each one as a SAC or TD3 model (by checking load_sac_agent 
and on failure load_td3_agent). If there's a config.json in the same subfolder, 
it prefers that, otherwise it uses the default `--agent_config` from the command line.

Usage example:
  python evaluate_elo.py \
      --model_folder ./models \
      --agent_config ./default_config.json \
      --env_mode NORMAL \
      --episodes_per_match 30
"""

import argparse
import os
import json
import numpy as np
import torch
import trueskill  # pip install trueskill
from pathlib import Path
import sys

# Adjust if needed so that TD3, etc. are importable.
sys.path.append("../")  

# import your agent classes
from DDQN.dqn_action_space import CustomActionSpace
from DDQN.DDQN import DoubleDuelingDQNAgent, DuelingDQNAgent
from DDQN.DQN import DoubleDQNAgent, TargetDQNAgent, DQNAgent
from TD3.src.td3 import TD3
from hockey_env import HockeyEnv, Mode, BasicOpponent
from sac import SACAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ddqn_agent(config_path, checkpoint_path, env, type=DoubleDuelingDQNAgent):
    dqn_classes = {
        "dqn": DQNAgent,
        "targ-dqn": TargetDQNAgent,
        "doub-dqn": DoubleDQNAgent,
        "duel-dqn": DuelingDQNAgent,
        "doub-duel-dqn": DoubleDuelingDQNAgent,
    }
    
    if type != DoubleDuelingDQNAgent:
        raise ValueError("Only DoubleDuelingDQNAgent is supported for now.")
    
    if "custactspc" in checkpoint_path:
        act_space = CustomActionSpace()
    else:
        act_space = env.discrete_action_space
    
    agent = type(
        env.observation_space,
        act_space,
        hidden_sizes=[512],
        hidden_sizes_A=[512, 512],
        hidden_sizes_V=[512, 512],
        use_torch=True,
    )
    agent.load_state(checkpoint_path)
    print(f"[DDQN] Loaded agent from checkpoint: {checkpoint_path}")
    return agent
    
    
    
def load_sac_agent(config_path, checkpoint_path, env):
    """
    Attempt to load an SACAgent from a checkpoint + config.

    If it fails (e.g. KeyError, structure mismatch), the caller can catch and
    attempt a different loader (like TD3).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "config" in checkpoint:
        config = checkpoint["config"]
        print(f"[SAC] Found config in checkpoint: {checkpoint_path}")
    else:
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"[SAC] Loaded agent config from file: {config_path}")

    # Possibly parse hidden_sizes, learn_alpha, etc.
    if isinstance(config.get("hidden_sizes_actor"), str):
        config["hidden_sizes_actor"] = list(map(int, config["hidden_sizes_actor"].split(",")))
    if isinstance(config.get("hidden_sizes_critic"), str):
        config["hidden_sizes_critic"] = list(map(int, config["hidden_sizes_critic"].split(",")))

    learn_alpha = config.get("learn_alpha", True)
    if isinstance(learn_alpha, str):
        learn_alpha = learn_alpha.lower() == "true"

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
        noise=config.get("noise", {
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

    agent.restore_full_state(checkpoint)
    return agent


def load_td3_agent(config_path, checkpoint_prefix, env):
    """
    Loads a TD3 agent from a JSON config and checkpoint prefix.
    The TD3 checkpoint is assumed to be saved as multiple files with the prefix, e.g.:
       - path/to/checkpoint_actor.pth
       - path/to/checkpoint_critic.pth
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    print("[TD3] Loaded TD3 configuration from:", config_path)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2
    max_action = env.action_space.high[0]

    agent = TD3(state_dim, action_dim, max_action, training_config=config)

    import copy
    agent.critic.load_state_dict(torch.load(checkpoint_prefix + "_critic.pth", map_location=device))
    agent.critic_optimizer.load_state_dict(torch.load(checkpoint_prefix + "_critic_optimizer.pth", map_location=device))
    agent.critic_target = copy.deepcopy(agent.critic)

    agent.actor.load_state_dict(torch.load(checkpoint_prefix + "_actor.pth", map_location=device))
    agent.actor_optimizer.load_state_dict(torch.load(checkpoint_prefix + "_actor_optimizer.pth", map_location=device))
    agent.actor_target = copy.deepcopy(agent.actor)

    if getattr(agent, 'use_rnd', False):
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


def evaluate_matchup(agentA, agentB, env, episodes=30):
    """
    Plays 'episodes' matches of agentA vs agentB.
    Returns (winsA, winsB, draws).
    """
    winsA = 0
    winsB = 0
    draws = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # agentA picks action
            # For consistency, let's define a method: agent.act(obs, eval_mode=True)
            actionA = agentA.act(obs)

            # agentB picks action
            if hasattr(env, "obs_agent_two"):
                obsB = env.obs_agent_two()
            else:
                obsB = obs
            actionB = agentB.act(obsB) if hasattr(agentB, "act") else np.zeros(env.action_space.shape[0]//2)
            
            full_action = np.hstack([actionA, actionB])
            obs, reward, done, trunc, info = env.step(full_action)
            done = done or trunc

        winner = info.get("winner", 0)
        if winner == 1:
            winsA += 1
        elif winner == -1:
            winsB += 1
        else:
            draws += 1
    return winsA, winsB, draws


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, required=True,
                        help="Folder containing subfolders with *.pth SAC or TD3 checkpoints.")
    parser.add_argument("--agent_config", type=str, required=True,
                        help="Fallback JSON config file if none in subfolder.")
    parser.add_argument("--env_mode", type=str, default="NORMAL",
                        help="HockeyEnv mode (NORMAL, TRAIN_SHOOTING, etc.)")
    parser.add_argument("--episodes_per_match", type=int, default=20,
                        help="Number of episodes to play per matchup.")
    return parser.parse_args()


def main():
    args = parse_args()
    mode = Mode[args.env_mode.upper()]
    env = HockeyEnv(mode=mode)
    match_results = []

    # 1) Recursively collect all .pth files in model_folder
    model_folder = Path(args.model_folder)
    
    # get all checkpoints that are NOT in a subfolder:
    checkpoints = sorted(model_folder.glob("*.pth"))
    
    # now find subfolders
    subfolders = [d for d in model_folder.iterdir() if d.is_dir()]
    # nor get only 1 checkpoint from each subfolder
    checkpoints_from_subfolders = []
    for d in subfolders:
        # get only 1 checkpoint from each subfolder
        checkpoints_from_subfolders.extend(sorted(d.glob("*.pth"))[:1])
    checkpoints.extend(checkpoints_from_subfolders)
    
    
    
    print(f"Found {len(checkpoints)} .pth checkpoint(s) under: {model_folder}")

    # 2) Prepare the "players" list
    players = []

    # Built-in "weak" / "strong" as well
    weak_opp = BasicOpponent(weak=True, keep_mode=env.keep_mode)
    players.append({
        "name": "weak_opponent",
        "agent": weak_opp,
        "rating": trueskill.Rating()
    })

    strong_opp = BasicOpponent(weak=False, keep_mode=env.keep_mode)
    players.append({
        "name": "strong_opponent",
        "agent": strong_opp,
        "rating": trueskill.Rating()
    })
    
    

    # 3) For each .pth file, attempt to load as SAC or TD3
    for ckpt in checkpoints:
        # e.g. subfolder "TD3_experiment1/checkpoint_episode_2000.pth"
        agent_name = f"{ckpt.parent.name}_{ckpt.stem}"
        print(f"Loading agent {agent_name} from {ckpt}")

        # Check if there's a config.json in the same folder as ckpt
        # If yes, prefer that. Otherwise fallback to the user-provided agent_config
        config_in_same_folder = ckpt.parent / "config.json"
        configflag = False
        if config_in_same_folder.exists():
            used_config = config_in_same_folder
            configflag = True
        else:
            used_config = Path(args.agent_config)

        # Try loading as SAC first
        if not configflag:
            tmp_agent = load_sac_agent(str(used_config), str(ckpt), env)
            print(f"  => Loaded as SAC successfully.")
        else:
            # If fails, try loading as TD3 (the checkpoint prefix is everything except "_actor.pth", etc.)
            # Typically your TD3 code expects a prefix, e.g. "my_ckpt" => "my_ckpt_actor.pth"
            # If your .pth file is "my_ckpt.pth" for everything, you'd need to adapt
            # For a typical approach, let's remove suffix "_actor/critic". But you have only one .pth?
            # If your TD3 saving approach uses multiple pth files, we might be missing them. 
            # We'll just do a fallback:
            # find the first dot
            prefix = str(ckpt)[:str(ckpt).find(".")] + ".pth"
            try:
                tmp_agent = load_td3_agent(str(used_config), prefix, env)
                print(f"  => Loaded as TD3 successfully.")
            except Exception as e_td3:
                # Both loading attempts failed
                print(f"Error: Could not load {ckpt} as SAC or TD3.\nSAC error: {e_sac}\nTD3 error: {e_td3}")
                continue

        # We require each agent to have an .act(obs) method for evaluation
        # If your agent is an SACAgent or TD3, we define a simple wrapper that calls act with no noise
        # (assuming your agent has .act(obs, add_noise=False) or .act(obs, eval_mode=True)).
        # We'll define it inline:
        class EvalWrapper:
            def __init__(self, raw_agent):
                self.raw_agent = raw_agent
            def act(self, obs):
                # If it's a SACAgent, we do raw_agent.act(obs, eval_mode=True)
                # If it's TD3, typically raw_agent.act(obs, add_noise=False)
                # We'll guess:
                if isinstance(self.raw_agent, SACAgent):
                    return self.raw_agent.act(obs, eval_mode=True)
                elif isinstance(self.raw_agent, TD3):
                    return self.raw_agent.act(obs, add_noise=False)
                else:
                    # unknown
                    return np.zeros(env.action_space.shape[0]//2, dtype=np.float32)

        final_agent_for_eval = EvalWrapper(tmp_agent)

        players.append({
            "name": agent_name,
            "agent": final_agent_for_eval,
            "rating": trueskill.Rating()
        })

    # 4) Round-robin: for each pair i<j, run episodes, update rating
    for i in range(len(players)):
        for j in range(i+1, len(players)):
            name_i = players[i]["name"]
            name_j = players[j]["name"]
            agent_i = players[i]["agent"]
            agent_j = players[j]["agent"]

            wins_i, wins_j, draws = evaluate_matchup(agent_i, agent_j, env, args.episodes_per_match)
            print(f"Match {name_i} vs {name_j} -> (wins_i={wins_i}, wins_j={wins_j}, draws={draws})")
            match_results.append({
                "agentA": name_i,
                "agentB": name_j,
                "winsA": wins_i,
                "winsB": wins_j,
                "draws": draws
            })

            # Summarize outcome in terms of "score" from each player's perspective
            if wins_i > wins_j:
                # i beats j
                players[i]["rating"], players[j]["rating"] = trueskill.rate_1vs1(players[i]["rating"], players[j]["rating"])
            elif wins_j > wins_i:
                # j beats i
                players[j]["rating"], players[i]["rating"] = trueskill.rate_1vs1(players[j]["rating"], players[i]["rating"])
            else:
                # draw
                players[i]["rating"], players[j]["rating"] = trueskill.rate_1vs1(players[i]["rating"], players[j]["rating"], drawn=True)

    env.close()

    # 5) Save match results
    import json
    with open("match_results.json", "w") as f:
        json.dump(match_results, f, indent=2)

    # 6) Sort and print final TrueSkill
    def trueskill_lcb(rating):
        return rating.mu - 3*rating.sigma

    # Sort by mu (descending)
    players.sort(key=lambda p: p["rating"].mu, reverse=True)

    print("\nFinal TrueSkill ratings (sorted by mu):")
    for p in players:
        r = p["rating"]
        lcb = trueskill_lcb(r)
        print(f"  {p['name']}: mu={r.mu:.2f}, sigma={r.sigma:.2f}, LCB={lcb:.2f}")

    # Also save final ratings to a file
    with open("final_ratings.txt", "w") as fr:
        fr.write("Final TrueSkill ratings:\n\n")
        for p in players:
            r = p["rating"]
            lcb = trueskill_lcb(r)
            fr.write(f"{p['name']}: mu={r.mu:.2f}, sigma={r.sigma:.2f}, LCB={lcb:.2f}\n")


if __name__ == "__main__":
    main()
