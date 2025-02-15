import argparse
import os
import sys
import uuid

import numpy as np
import torch
from comprl.client import Agent, launch_client

root_dir = os.path.dirname(os.path.abspath("../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import hockey.hockey_env as h_env
from DDQN.action_space import CustomActionSpace
from DDQN.DQN import DQNAgent, TargetDQNAgent, DoubleDQNAgent
from DDQN.DDQN import DuelingDQNAgent, DoubleDuelingDQNAgent


class ClientAgent(Agent):
    def __init__(self, agent: DQNAgent, env: h_env.HockeyEnv, action_space_type: str):
        self.agent = agent
        self.env = env
        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if action_space_type == "default":
            self.discrete_to_continuous = self.env.discrete_to_continous_action
        elif action_space_type == "custom":
            self.discrete_to_continuous = CustomActionSpace.discrete_to_continuous
        else:
            raise ValueError(f"Invalid action space type: {action_space_type}")

    def np2gpu(self, data: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(data).float().to(self.train_device)

    def get_step(self, obs: list[float]) -> list[float]:
        obs = self.np2gpu(np.array(obs))
        act_discr = self.agent.act_torch(obs)
        act = self.discrete_to_continuous(act_discr)

        if type(act) is np.ndarray:
            act = act.tolist()

        return act
    
    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")
    
    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_type", type=str, help="Type of the agent to train",
        choices=["dqn", "targ-dqn", "doub-dqn", "duel-dqn", "doub-duel-dqn"])
    parser.add_argument("model_path", type=str, default=None,
        help="Path to the model file to load")
    parser.add_argument("action_space", type=str, default="custom",
        help="Action space to use", choices=["default", "custom"])
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[512],
        help="Hidden layer sizes for the Q function")
    parser.add_argument("--hidden-sizes-A", type=int, nargs="+", default=[512, 512],
        help="Hidden layer sizes for the advantage stream in Dueling DQN")
    parser.add_argument("--hidden-sizes-V", type=int, nargs="+", default=[512, 512],
        help="Hidden layer sizes for the value stream in Dueling DQN")
    parser.add_argument("--use-numpy", action="store_true",
        help="Use NumPy functionalities for training")

    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    match args.agent_type:
        case "dqn":
            agent_class = DQNAgent
        case "targ-dqn":
            agent_class = TargetDQNAgent
        case "doub-dqn":
            agent_class = DoubleDQNAgent
        case "duel-dqn":
            agent_class = DuelingDQNAgent
        case "doub-duel-dqn":
            agent_class = DoubleDuelingDQNAgent
        case _:
            raise ValueError(f"Invalid agent type: {args.agent_type}")

    env = h_env.HockeyEnv()

    match args.action_space:
        case "default":
            action_space = env.discrete_action_space
        case "custom":
            action_space = CustomActionSpace()
        case _:
            raise ValueError(f"Invalid action space: {args.action_space}")

    agent_player = agent_class(
        env.observation_space,
        action_space,
        hidden_sizes=args.hidden_sizes,
        hidden_sizes_A=args.hidden_sizes_A,
        hidden_sizes_V=args.hidden_sizes_V,
        use_torch=(not args.use_numpy)
    )

    agent_player.load_state(args.model_path)

    # Create the client agent and return it.
    agent = ClientAgent(agent_player, env, action_space_type=args.action_space)

    return agent


if __name__ == "__main__":
    launch_client(initialize_agent)