import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Adds current directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Adds parent directory
import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from noise import *
from memory import ReplayMemory, PrioritizedExperienceReplay, EREPrioritizedExperienceReplay
from policies import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)


def select_loss_function(loss_type):
    """
    Select and return the appropriate loss function based on the input string.

    Args:
        loss_type (str): Type of loss function to use

    Returns:
        callable: Loss function to be used for critic training
    """
    loss_functions = {
        "mse": F.mse_loss,
        "huber": F.smooth_l1_loss,
        "mae": F.l1_loss,
        "mse_weighted": lambda input, target: torch.mean(
            F.mse_loss(input, target, reduction="none") * 1.0
        ),
    }

    return loss_functions.get(loss_type.lower(), F.mse_loss)


class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible"""

    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)


class SACAgent:
    def __init__(
        self, observation_space, action_space, loss_fn=F.mse_loss, **userconfig
    ):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace(
                "Observation space {} incompatible ".format(observation_space)
            )
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace("Action space {} incompatible ".format(action_space))

        self._observation_space = observation_space
        self._action_space = action_space            
        self._action_dim = action_space.shape[0]
        self.critic_loss_fn = loss_fn

        # Default configuration
        self._config = {
            "discount": 0.99,
            "buffer_size": int(1e6),
            "batch_size": 256,
            "learning_rate_actor": 3e-4,
            "learning_rate_critic": 3e-4,
            "hidden_sizes_actor": [256, 256],
            "hidden_sizes_critic": [256, 256],
            "tau": 0.005,  # Polyak update
            "alpha": 0.2,  # Temperature parameter
            "learn_alpha": True,  # Whether to learn temperature parameter
            "update_every": 1,
            "epsilon": 1e-6,
            # PER parameters
            "use_per": False,
            "per_alpha": 0.6,  # beta_1 in PER paper
            "per_beta": 0.4,  # beta_2 in PER paper
            "per_beta_increment": 0.001,
            # ERE parameters
            "use_ere": False,
            "ere_eta0": 0.996,  # Initial ERE decay rate
            "ere_etaT": 1.0,  # Final ERE decay rate
            "ere_c_k_min": 2500,  # Minimum ERE buffer size
            # Noise
            "noise": {
                "type": "normal",
                "sigma": 0.1,
                "theta": 0.15,
                "dt": 1e-2,
                "beta": 1.0,
                "seq_len": 1000,
            },
        }
        if "noise" in userconfig:
            self._config["noise"].update(userconfig.pop("noise"))
        # Handle remaining non-noise parameters
        self._config.update(userconfig)

        # Initialize networks
        self.actor = Actor(
            self._observation_space,
            self._action_dim,
            self._config["hidden_sizes_actor"],
            self._config["learning_rate_actor"],
            action_space=action_space,
            device=device,
            epsilon=self._config["epsilon"],
            noise_config=self._config["noise"],
        )

        # Two Q-functions to mitigate positive bias in policy improvement
        self.critic1 = Critic(
            self._observation_space,
            self._action_dim,
            self._config["hidden_sizes_critic"],
            self._config["learning_rate_critic"],
        )
        self.critic2 = Critic(
            self._observation_space,
            self._action_dim,
            self._config["hidden_sizes_critic"],
            self._config["learning_rate_critic"],
        )

        # Target networks
        self.critic1_target = Critic(
            self._observation_space,
            self._action_dim,
            self._config["hidden_sizes_critic"],
            self._config["learning_rate_critic"],
        )
        self.critic2_target = Critic(
            self._observation_space,
            self._action_dim,
            self._config["hidden_sizes_critic"],
            self._config["learning_rate_critic"],
        )

        # Copy parameters to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Initialize alpha (entropy temperature)
        if self._config["learn_alpha"]:
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=self._config["learning_rate_actor"]
            )
        else:
            self.alpha = torch.tensor(self._config["alpha"])

        # Initialize replay buffer (PER or standard)
        if self._config["use_per"]:
            if self._config["use_ere"]:
                self.buffer = EREPrioritizedExperienceReplay(
                    max_size=self._config["buffer_size"],
                    beta_1=self._config["per_alpha"],
                    beta_2=self._config["per_beta"],
                    beta_increment=self._config["per_beta_increment"],
                    epsilon=self._config["epsilon"],
                    eta_0=self._config["ere_eta0"],
                    eta_T=self._config["ere_etaT"],
                    c_k_min=self._config["ere_c_k_min"],
                )
            else:
                self.buffer = PrioritizedExperienceReplay(
                    max_size=self._config["buffer_size"],
                    beta_1=self._config["per_alpha"],
                    beta_2=self._config["per_beta"],
                    beta_increment=self._config["per_beta_increment"],
                    epsilon=self._config["epsilon"]
                )
        else:
            self.buffer = ReplayMemory(max_size=self._config["buffer_size"])

        self.train_iter = 0
        self.K = 0  # Track total steps for ERE

        # Move networks to device
        self.actor.to(device)
        self.critic1.to(device)
        self.critic2.to(device)
        self.critic1_target.to(device)
        self.critic2_target.to(device)

    def reset_noise(self):
        """Reset the noise process if it has a reset method"""
        if hasattr(self.actor.noise, "reset"):
            self.actor.noise.reset()

    def act(self, observation, eval_mode=False):
        observation = torch.FloatTensor(observation).unsqueeze(0).to(device)

        with torch.no_grad():
            if eval_mode:
                action_mean, _ = self.actor(observation)
                action = (
                    torch.tanh(action_mean) * self.actor.action_scale
                    + self.actor.action_bias
                )
            else:
                action, _ = self.actor.sample(observation)

        return action.cpu().numpy()[0]

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def train(self, iter_fit=32):
        losses = []

        for k in range(iter_fit):
            # Sample from replay buffer with PER/ERE if enabled
            if isinstance(self.buffer, PrioritizedExperienceReplay):
                experiences, indices, weights = self.buffer.sample(
                    batch=self._config["batch_size"], step=k, total_steps=self.K
                )
                state = torch.FloatTensor(experiences[0]).to(device)  # Already stacked states
                action = torch.FloatTensor(experiences[1]).to(device)  # Already stacked actions
                reward = torch.FloatTensor(experiences[2][:, None]).to(device)  # Add dimension for rewards
                next_state = torch.FloatTensor(experiences[3]).to(device)  # Already stacked next_states
                done = torch.FloatTensor(experiences[4][:, None]).to(device)  # Add dimension for dones

                weights = torch.FloatTensor(weights).to(device)
            else:
                batch = self.buffer.sample(batch=self._config["batch_size"])
                state = torch.FloatTensor(np.stack(batch[:, 0])).to(device)
                action = torch.FloatTensor(np.stack(batch[:, 1])).to(device)
                reward = torch.FloatTensor(np.stack(batch[:, 2])[:, None]).to(device)
                next_state = torch.FloatTensor(np.stack(batch[:, 3])).to(device)
                done = torch.FloatTensor(np.stack(batch[:, 4])[:, None]).to(device)
                weights = torch.ones_like(reward).to(device)

            with torch.no_grad():
                # Sample next action and compute Q-target
                next_action, next_log_prob = self.actor.sample(next_state)

                q1_target = self.critic1_target(next_state, next_action)
                q2_target = self.critic2_target(next_state, next_action)
                q_target = torch.min(q1_target, q2_target)

                # Compute target with entropy
                value_target = q_target - self.alpha * next_log_prob
                q_backup = reward + (1 - done) * self._config["discount"] * value_target

            # Update critics
            q1 = self.critic1(state, action)
            q2 = self.critic2(state, action)

            # Compute TD errors for priority updates
            td_error1 = q_backup.detach() - q1
            td_error2 = q_backup.detach() - q2

            # Compute critic losses with importance sampling weights
            critic1_loss = (td_error1.pow(2) * weights).mean()
            critic2_loss = (td_error2.pow(2) * weights).mean()

            self.critic1.optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1.optimizer.step()

            self.critic2.optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2.optimizer.step()

            # Update PER priorities if using PER
            if isinstance(self.buffer, PrioritizedExperienceReplay):
                new_priorities = abs(
                    ((td_error1 + td_error2) / 2.0 + self._config["epsilon"])
                    .detach()
                    .cpu()
                    .numpy()
                )
                self.buffer.update_priorities(indices, new_priorities)

            # Update actor (using importance sampling weights)
            action_new, log_prob = self.actor.sample(state)
            q1_new = self.critic1(state, action_new)
            q2_new = self.critic2(state, action_new)
            q_new = torch.min(q1_new, q2_new)

            actor_loss = (self.alpha * log_prob - q_new).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update alpha if necessary (using importance sampling weights)
            if self._config["learn_alpha"]:
                alpha_loss = -(
                    self.log_alpha * (log_prob + self.target_entropy).detach()
                ).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = self.log_alpha.exp()

            # Update target networks
            if self.train_iter % self._config["update_every"] == 0:
                for param, target_param in zip(
                    self.critic1.parameters(), self.critic1_target.parameters()
                ):
                    target_param.data.copy_(
                        self._config["tau"] * param.data
                        + (1 - self._config["tau"]) * target_param.data
                    )
                for param, target_param in zip(
                    self.critic2.parameters(), self.critic2_target.parameters()
                ):
                    target_param.data.copy_(
                        self._config["tau"] * param.data
                        + (1 - self._config["tau"]) * target_param.data
                    )

            self.train_iter += 1
            losses.append((critic1_loss.item(), critic2_loss.item(), actor_loss.item()))

        return losses

    def state(self):
        return (
            self.actor.state_dict(),
            self.critic1.state_dict(),
            self.critic2.state_dict(),
        )

    def restore_state(self, state):
        self.actor.load_state_dict(state[0])
        self.critic1.load_state_dict(state[1])
        self.critic2.load_state_dict(state[2])
        self.critic1_target.load_state_dict(state[1])
        self.critic2_target.load_state_dict(state[2])
