#!/usr/bin/env python
"""
Evaluate multiple SAC checkpoints + built-in opponents in a round-robin fashion
and track their ratings using TrueSkill (similar to an Elo system).

Usage example:
  python evaluate_elo.py \
      --model_folder ./models_sac \
      --agent_config ./config_sac.json \
      --env_mode NORMAL \
      --episodes_per_match 30

This will:
  1) Load every *.pth file in ./models_sac as a separate SAC agent.
  2) Add two built-in opponents: "weak" and "strong".
  3) Play each agent vs. each other agent for 30 episodes. 
  4) Update TrueSkill ratings after each match. 
  5) Print the final rating for each agent, including LCB = mu - 3*sigma.
"""

import argparse
import os
import json
import numpy as np
import torch
import trueskill  # pip install trueskill
from pathlib import Path

# If your "arena code" is in the same file, you can copy it here or import it.
# We'll assume you have the same environment code as in your snippet.
# Just be sure that `HockeyEnv`, `BasicOpponent`, `SACAgent`, etc. are in the path.
from hockey_env import HockeyEnv, Mode, BasicOpponent
from sac import SACAgent
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sac_agent(config_path, checkpoint_path, env):
    # The same function you have in your arena code:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "config" in checkpoint:
        config = checkpoint["config"]
        print(f"Loaded agent config from checkpoint: {checkpoint_path}")
    else:
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"Loaded agent config from file: {config_path}")

    if isinstance(checkpoint, tuple):
        # handle tuple case
        pass

    # Possibly parse hidden_sizes, learn_alpha, etc. ...
    if isinstance(config["hidden_sizes_actor"], str):
        config["hidden_sizes_actor"] = list(map(int, config["hidden_sizes_actor"].split(",")))
    if isinstance(config["hidden_sizes_critic"], str):
        config["hidden_sizes_critic"] = list(map(int, config["hidden_sizes_critic"].split(",")))

    learn_alpha = config["learn_alpha"]
    if isinstance(learn_alpha, str):
        learn_alpha = (learn_alpha.lower() == "true")

    agent = SACAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        discount=config.get("discount", 0.99),
        buffer_size=config.get("buffer_size", int(1e6)),
        learning_rate_actor=config.get("learning_rate_actor", 3e-4),
        learning_rate_critic=config.get("learning_rate_critic", 3e-4),
        # ...
        hidden_sizes_actor=config["hidden_sizes_actor"],
        hidden_sizes_critic=config["hidden_sizes_critic"],
        learn_alpha=learn_alpha,
        alpha=config.get("alpha", 0.2),
        control_half=True
    )
    agent.restore_full_state(checkpoint)
    return agent


def evaluate_matchup(agentA, agentB, env, episodes=30):
    """
    Plays 'episodes' matches of agentA vs agentB.
    Returns (winsA, winsB, draws).
    """
    # If agentB is BasicOpponent, it has an .act() method that is already consistent.
    # If agentB is a "weak" or "strong" string, we handle that outside or convert to BasicOpponent.

    winsA = 0
    winsB = 0
    draws = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # agentA picks action
            actionA = agentA.act(obs, eval_mode=True)

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
                        help="Folder containing *.pth SAC checkpoints")
    parser.add_argument("--agent_config", type=str, required=True,
                        help="The JSON config file that all those models share (or a superset).")
    parser.add_argument("--env_mode", type=str, default="NORMAL",
                        help="HockeyEnv mode (NORMAL, TRAIN_SHOOTING, etc.)")
    parser.add_argument("--episodes_per_match", type=int, default=20,
                        help="Number of episodes to play per matchup.")
    return parser.parse_args()


def main():
    args = parse_args()
    mode = Mode[args.env_mode.upper()]
    env = HockeyEnv(mode=mode)

    # 1) Collect all the .pth files in model_folder
    model_folder = Path(args.model_folder)
    checkpoints = sorted(model_folder.glob("*.pth"))
    print(f"Found {len(checkpoints)} checkpoint(s) in {model_folder}:")

    # 2) Create a list of "players"
    # We'll store dicts like {"name": str, "agent": <SACAgent or BasicOpponent>, "rating": Rating()}
    players = []

    # Add the "weak" built-in opponent
    weak_opp = BasicOpponent(weak=True, keep_mode=env.keep_mode)
    players.append({
        "name": "weak_opponent",
        "agent": weak_opp,
        "rating": trueskill.Rating()
    })

    # Add the "strong" built-in opponent
    strong_opp = BasicOpponent(weak=False, keep_mode=env.keep_mode)
    players.append({
        "name": "strong_opponent",
        "agent": strong_opp,
        "rating": trueskill.Rating()
    })

    # For each .pth file, load the agent
    for ckpt in checkpoints:
        agent_name = ckpt.stem  # e.g. "checkpoint_episode_1000"
        print(f"  - Loading agent {agent_name} from {ckpt}")
        agent = load_sac_agent(args.agent_config, str(ckpt), env)
        players.append({
            "name": agent_name,
            "agent": agent,
            "rating": trueskill.Rating()
        })

    # 3) Round-robin: for each pair i<j, run episodes, update rating
    for i in range(len(players)):
        for j in range(i+1, len(players)):
            name_i = players[i]["name"]
            name_j = players[j]["name"]
            agent_i = players[i]["agent"]
            agent_j = players[j]["agent"]

            wins_i, wins_j, draws = evaluate_matchup(agent_i, agent_j, env, args.episodes_per_match)
            print(f"Match {name_i} vs {name_j} -> (wins_i={wins_i}, wins_j={wins_j}, draws={draws})")

            # Summarize outcome in terms of "score" from each player's perspective
            # We'll do something like:
            #   agent i's points = wins_i + draws*0.5
            #   agent j's points = wins_j + draws*0.5
            # But TrueSkill requires a single match result (1 vs 2). 
            # A quick hack is to treat "who got more wins overall" as the match outcome. 
            # If we want to treat draws as well, we can do repeated rating updates, or partial updates. 
            # For simplicity, let's say:
            
            if wins_i > wins_j:
                # i beats j
                players[i]["rating"], players[j]["rating"] = trueskill.rate_1vs1(players[i]["rating"], players[j]["rating"])
            elif wins_j > wins_i:
                # j beats i
                players[j]["rating"], players[i]["rating"] = trueskill.rate_1vs1(players[j]["rating"], players[i]["rating"])
            else:
                # draw
                players[i]["rating"], players[j]["rating"] = trueskill.rate_1vs1(players[i]["rating"], players[j]["rating"], drawn=True)

    # 4) Print final ratings
    # We can sort them by their conservative rating (mu - 3*sigma).
    def trueskill_lcb(rating):
        return rating.mu - 3*rating.sigma

    players.sort(key=lambda p: trueskill_lcb(p["rating"]), reverse=True)
    print("\nFinal TrueSkill ratings (sorted by mu - 3*sigma):")
    for p in players:
        r = p["rating"]
        print(f"  {p['name']}: mu={r.mu:.2f}, sigma={r.sigma:.2f}, LCB={r.mu - 3*r.sigma:.2f}")


if __name__ == "__main__":
    main()
