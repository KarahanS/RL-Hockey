# td3.py

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .actor import Actor
from .critic import Critic
from .rnd import RND  # Import the updated RND class
from .pink_noise import PinkNoise
from .utils import OUNoise  # Ensure this is correctly implemented/imported

class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        training_config,  # Unified training configuration
    ):  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store action dimension
        self.action_dim = action_dim

        # Initialize Actor
        self.actor = Actor(
            state_dim,
            action_dim,
            max_action,
            use_layer_norm=training_config.get("use_layer_norm", False),
            ln_eps=training_config.get("ln_eps", 1e-5)
        ).to(self.device)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        # Initialize Critic
        self.critic = Critic(
            state_dim,
            action_dim,
            use_layer_norm=training_config.get("use_layer_norm", False),
            ln_eps=training_config.get("ln_eps", 1e-5)
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        # Initialize RND if enabled
        self.use_rnd = training_config.get("use_rnd", False)
        if self.use_rnd:
            self.rnd = RND(state_dim, self.device, hidden_dim=training_config.get("rnd_hidden_dim", 128), lr=training_config.get("rnd_lr", 1e-4))
            self.rnd_weight = training_config.get("rnd_weight", 1.0)  # Scaling factor for intrinsic rewards

        self.max_action = max_action
        self.discount = training_config.get("discount", 0.99)
        self.tau = training_config.get("tau", 0.005)
        self.policy_noise = training_config.get("policy_noise", 0.2)
        self.noise_clip = training_config.get("noise_clip", 0.5)
        self.policy_freq = training_config.get("policy_freq", 2)

        # Initialize noise generator based on configuration
        self.noise_type = training_config.get("expl_noise_type", "gaussian").lower()
        self.expl_noise_scale = training_config.get("expl_noise", 0.1)

        if self.noise_type == "pink":
            self.pink_noise = PinkNoise(
                action_dim=action_dim,
                max_steps=training_config.get("max_episode_steps", 1000),
                exponent=training_config.get("pink_noise_params", {}).get("exponent", 1.0),
                fmin=training_config.get("pink_noise_params", {}).get("fmin", 0.0)
            )
        elif self.noise_type == "ou":
            self.ou_noise = OUNoise(
                action_dim=action_dim,
                mu=training_config.get("ou_noise_params", {}).get("mu", 0.0),
                theta=training_config.get("ou_noise_params", {}).get("theta", 0.15),
                sigma=training_config.get("ou_noise_params", {}).get("sigma", 0.2)
            )
        else:
            self.pink_noise = None
            self.ou_noise = None  # No additional noise generator needed for Gaussian

        self.total_it = 0

    def act(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            if self.noise_type == "gaussian":
                noise = np.random.normal(0, self.expl_noise_scale, size=self.action_dim)
                action = (action + noise).clip(-self.max_action, self.max_action)
            elif self.noise_type == "pink" and self.pink_noise is not None:
                noise = self.pink_noise.get_noise() * self.expl_noise_scale
                action = (action + noise).clip(-self.max_action, self.max_action)
            elif self.noise_type == "ou" and self.ou_noise is not None:
                noise = self.ou_noise.sample() * self.expl_noise_scale
                action = (action + noise).clip(-self.max_action, self.max_action)
        return action

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Ensure data types
        state = state.float().to(self.device)
        action = action.float().to(self.device)
        next_state = next_state.float().to(self.device)
        reward = reward.float().to(self.device)
        not_done = not_done.float().to(self.device)

        if self.use_rnd:
            # Compute intrinsic rewards for the batch
            intrinsic_rewards = self.rnd.compute_intrinsic_reward(state)
            # Scale intrinsic rewards
            intrinsic_rewards = intrinsic_rewards * self.rnd_weight
            # Combine extrinsic and intrinsic rewards
            total_reward = reward + intrinsic_rewards
        else:
            total_reward = reward

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = total_reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None  # Initialize actor_loss
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update RND predictor network if enabled
        if self.use_rnd:
            rnd_loss = self.rnd.update_predictor(state)

        return critic_loss.item(), actor_loss.item() if actor_loss is not None else None

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer.pth")
        
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pth")
        
        if self.use_rnd:
            torch.save(self.rnd.target_network.state_dict(), filename + "_rnd_target.pth")
            torch.save(self.rnd.predictor_network.state_dict(), filename + "_rnd_predictor.pth")
            torch.save(self.rnd.optimizer.state_dict(), filename + "_rnd_optimizer.pth")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", weights_only=True))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pth", weights_only=True))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor.pth", weights_only=True))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pth", weights_only=True))
        self.actor_target = copy.deepcopy(self.actor)
        
        if self.use_rnd:
            self.rnd.target_network.load_state_dict(torch.load(filename + "_rnd_target.pth", weights_only=True))
            self.rnd.predictor_network.load_state_dict(torch.load(filename + "_rnd_predictor.pth", weights_only=True))
            self.rnd.optimizer.load_state_dict(torch.load(filename + "_rnd_optimizer.pth", weights_only=True))