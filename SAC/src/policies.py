import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Adds current directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Adds parent directory
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
from typing import Optional, Tuple, Union
from numpy.typing import DTypeLike
from torch import nn
import torch
from feedforward import FeedForward
from noise import *

class CNNEncoder(nn.Module):
    def __init__(self, observation_shape: Tuple):
        super().__init__()
        assert len(observation_shape) == 3, "Image input must have shape (H, W, C)"
        in_channels = observation_shape[-1]  # Channels last format
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
    def get_feature_size(self, observation_shape: Tuple) -> int:
        # Handle channels-last format (H, W, C) -> (C, H, W)
        C, H, W = observation_shape[-1], observation_shape[0], observation_shape[1]
        test_input = torch.zeros(1, C, H, W)
        with torch.no_grad():
            output = self.conv(test_input)
        return output.shape[1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle channels-last format (B, H, W, C) -> (B, C, H, W)
        if len(x.shape) == 3:  # Single image (H, W, C)
            x = x.unsqueeze(0)  # Add batch dimension
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        # Normalize input
        x = x / 255.0
        return self.conv(x)
    

class Actor(FeedForward):
    def __init__(
        self,
        observation_space,
        action_dim,
        hidden_sizes=[256, 256],
        learning_rate=3e-4,
        action_space=None,
        device="cpu",
        epsilon=1e-6,
        noise_config=None
    ):
        
        self.is_image_obs = len(observation_space.shape) == 3
        
        if self.is_image_obs:
            input_size = 1  # temporary
        else:
            input_size = observation_space.shape[0]
            
        super().__init__(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=action_dim * 2,  # Mean and log_std for each action
            hidden_activation=nn.ReLU(),
            output_activation=None,
        )
        if self.is_image_obs:
            self.cnn = CNNEncoder(observation_space.shape)
            feature_dim = self.cnn.get_feature_size(observation_space.shape)
            self.network[0] = nn.Linear(feature_dim, hidden_sizes[0])

        self.epsilon = epsilon
        self.action_bias = None
        self.action_dim = action_dim
        self.device = device
        if self.is_image_obs:
            self.optimizer = torch.optim.Adam(
                list(self.cnn.parameters()) + list(self.network.parameters()),
                lr=learning_rate
            )
        else:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

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

        noise_type = noise_config["type"]

        # Set action noise
        if noise_type == "normal":
            self.noise = NormalActionNoise(
                mean=np.zeros(action_dim),
                sigma=np.ones(action_dim) * noise_config["sigma"],
            )
        elif noise_type == "ornstein":
            self.noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(action_dim),
                sigma=np.ones(action_dim) * noise_config["sigma"],
                theta=noise_config["theta"],
                dt=noise_config["dt"],
            )
        elif noise_type == "colored":
            self.noise = ColoredActionNoise(
                beta=noise_config["beta"],
                sigma=noise_config["sigma"],
                seq_len=noise_config["seq_len"],
                action_dim=action_dim,
            )
        elif noise_type == "pink":
            self.noise = PinkActionNoise(
                sigma=noise_config["sigma"],
                seq_len=noise_config["seq_len"],
                action_dim=action_dim,
            )

    def forward(self, obs):
        if self.is_image_obs:
            features = self.cnn(obs)
        else:
            features = obs
        output = super().forward(features)
        mean, log_std = torch.chunk(output, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 10)
        return mean, log_std

    def sample(self, obs, use_exploration_noise: bool = False):
        """
        Sample an action given an observation.

        Args:
            obs: The input observation.
            use_exploration_noise: If True, adds extra exploration noise to the final action.
                                This extra noise is applied after computing the differentiable
                                part of the policy (for environment interaction only).

        Returns:
            action: The final (possibly clipped) action.
            log_prob: The log probability of the action before adding extra exploration noise.
        """
        # USE EXPLORATIVE NOISE ONLY DURING ROLL-OUTS (NOT DURING GRADIENT STEPS)
        # This function is used only during training (not evaluation, so don't worry about that)
        
        # --- Differentiable Reparameterization for Training ---
        # Compute mean and log_std from the network
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # Use standard Gaussian noise for the reparameterization trick
        noise = torch.randn_like(mean)
        x_t = mean + std * noise

        # Apply tanh squashing and scale to the action space
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # --- Compute Log Probability ---
        # Calculate the Gaussian log probability for x_t under N(mean, std)
        log_prob = -0.5 * (((x_t - mean) / (std + self.epsilon)) ** 2 +
                        2 * log_std +
                        np.log(2 * np.pi))
        # Apply correction for tanh squashing (Jacobian adjustment)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # --- Optional: Add Extra Exploration Noise ---
        # IMPORTANT: This extra noise is added for acting in the environment and is not used
        # in the gradient computation (log_prob remains based on the reparameterized action).
        if use_exploration_noise:
            extra_noise = torch.FloatTensor(self.noise()).to(self.device)
            action = action + extra_noise

        # --- Clip the Final Action ---
        # Calculate the valid bounds based on action_scale and action_bias:
        # For a tanh output in [-1,1], action = tanh_output * action_scale + action_bias is
        # naturally within [action_bias - action_scale, action_bias + action_scale].
        min_action = self.action_bias - self.action_scale # -1
        max_action = self.action_bias + self.action_scale # +1
        action = torch.clamp(action, min=min_action, max=max_action)

        return action, log_prob

    
    def get_state(self):
        """
        Get the entire internal state for checkpointing:
         - model (CNN + MLP) parameters
         - optimizer parameters
         - noise state (if implemented)
        """
        state = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # Noise state (if the noise class has get_state())
            "noise_state": self.noise.get_state() if hasattr(self.noise, "get_state") else None,
        }
        return state

    def set_state(self, state: dict):
        """
        Restore the internal state from a checkpoint dictionary.

        If any keys (like "optimizer_state_dict" or "noise_state") are not present,
        we skip them so that you can load model weights only.
        """
        # 1. Load the model weights (mandatory for inference)
        if "model_state_dict" in state:
            self.load_state_dict(state["model_state_dict"])
        else:
            print("[Actor] No 'model_state_dict' found in the checkpoint. Skipping model weights loading.")

        # 2. Load the optimizer state (only if provided)
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        else:
            print("[Actor] No 'optimizer_state_dict' found in the checkpoint. Skipping optimizer state loading.")

        # 3. Load the noise state (only if provided)
        noise_state = state.get("noise_state", None)
        if noise_state is not None and hasattr(self.noise, "set_state"):
            self.noise.set_state(noise_state)
        else:
            print("[Actor] No 'noise_state' found in the checkpoint or noise has no set_state(). Skipping noise state.")


class Critic(FeedForward):
    def __init__(
        self, observation_space, action_dim, hidden_sizes=[256, 256], learning_rate=3e-4
    ):
        # Check if we're dealing with image observations
        self.is_image_obs = len(observation_space.shape) == 3
        
        if self.is_image_obs:
            input_size = 1 # temporary
        else:
            input_size = observation_space.shape[0]
            
        super().__init__(
            input_size=input_size  + action_dim,
            hidden_sizes=hidden_sizes,
            output_size=1,
            hidden_activation=nn.ReLU(),
            output_activation=None,
        )
        if self.is_image_obs:
            self.cnn = CNNEncoder(observation_space.shape)
            feature_dim = self.cnn.get_feature_size(observation_space.shape)
            self.network[0] = nn.Linear(feature_dim + action_dim, hidden_sizes[0])
            
        if self.is_image_obs:
            self.optimizer = torch.optim.Adam(
                list(self.cnn.parameters()) + list(self.network.parameters()),
                lr=learning_rate
            )
        else:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def forward(self, obs, action):
        if self.is_image_obs:
            features = self.cnn(obs)
        else:
            features = obs
            
        x = torch.cat([features, action], dim=1)
        return super().forward(x)

    def get_state(self):
        """
        Get the entire internal state for checkpointing:
         - model (CNN + MLP) parameters
         - optimizer parameters
        """
        return {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def set_state(self, state: dict):
        """
        Restore the internal state from a checkpoint dictionary.

        If no optimizer_state_dict is found, we skip loading that part.
        """
        # 1. Load the model weights
        if "model_state_dict" in state:
            self.load_state_dict(state["model_state_dict"])
        else:
            print("[Critic] No 'model_state_dict' found in the checkpoint. Skipping model weights loading.")

        # 2. Load the optimizer state (only if provided)
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        else:
            print("[Critic] No 'optimizer_state_dict' found in the checkpoint. Skipping optimizer state loading.")

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
