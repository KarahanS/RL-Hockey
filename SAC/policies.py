from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
from numpy.typing import DTypeLike
from torch import nn
import torch
from feedforward import FeedForward
from noise import *


class Actor(FeedForward):
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_sizes=[256, 256],
        learning_rate=3e-4,
        action_space=None,
        device="cpu",
        epsilon=1e-6,
        noise_config=None,
    ):
        super().__init__(
            input_size=observation_dim,
            hidden_sizes=hidden_sizes,
            output_size=action_dim * 2,  # Mean and log_std for each action
            hidden_activation=nn.ReLU(),
            output_activation=None,
        )

        self.epsilon = epsilon
        self.action_bias = None
        self.action_dim = action_dim
        self.device = device
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Set up action scaling
        if action_space is None:
            self.action_scale = torch.ones(self.action_dim).to(device)
            self.action_bias = torch.zeros(self.action_dim).to(device)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2
            ).to(device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2
            ).to(device)

        noise_type = noise_config["noise_type"]

        # Set action noise
        if noise_type == "normal":
            self.noise = NormalActionNoise(
                mean=np.zeros(action_dim),
                sigma=np.ones(action_dim) * noise_config["noise_sigma"],
            )
        elif noise_type == "ornstein":
            self.noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(action_dim),
                sigma=np.ones(action_dim) * noise_config["noise_sigma"],
                theta=noise_config["noise_theta"],
                dt=noise_config["noise_dt"],
            )
        elif noise_type == "colored":
            self.noise = ColoredActionNoise(
                beta=noise_config["noise_beta"],
                sigma=noise_config["noise_sigma"],
                seq_len=noise_config["noise_seq_len"],
                action_dim=action_dim,
            )
        elif noise_type == "pink":
            self.noise = PinkActionNoise(
                sigma=noise_config["noise_sigma"],
                seq_len=noise_config["noise_seq_len"],
                action_dim=action_dim,
            )

    def forward(self, obs):
        output = super().forward(obs)
        mean, log_std = torch.chunk(output, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 10)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # regardless of what noise we are using, we always apply the following conversion:
        # x_t = mean + std * noise
        # y_t = tanh(x_t) (tanh squashing)
        # action = y_t * self.action_scale + self.action_bias

        noise = torch.FloatTensor(self.noise()).to(self.device)
        x_t = mean + std * noise

        # Squash using tanh
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # Calculate log probability
        log_prob = (
            -((x_t - mean) ** 2) / (2 * std**2) - log_std - np.log(np.sqrt(2 * np.pi))
        )
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class Critic(FeedForward):
    def __init__(
        self, observation_dim, action_dim, hidden_sizes=[256, 256], learning_rate=3e-4
    ):
        super().__init__(
            input_size=observation_dim + action_dim,
            hidden_sizes=hidden_sizes,
            output_size=1,
            hidden_activation=nn.ReLU(),
            output_activation=None,
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        return super().forward(x)


# Example usage:
"""
# Initialize actor and critic
actor = Actor(observation_dim=4, action_dim=2, device=device)
critic = Critic(observation_dim=4, action_dim=2)

# Use Gaussian noise (default)
actor.action_dist = GaussianDistribution(action_dim=2, epsilon=1e-6)

# Use OU noise
actor.action_dist = OUDistribution(
    action_dim=2,
    device=device,
    theta=0.15,
    sigma=0.2,
    dt=1e-2
)

# For colored noise, you can use the external library:
# from pink import PinkNoiseDist
# actor.action_dist = PinkNoiseDist(seq_len, action_dim, rng=rng)
"""
