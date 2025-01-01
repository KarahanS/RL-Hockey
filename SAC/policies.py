from torch import nn
from feedforward import FeedForward
import torch


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
    ):
        # The actor outputs mean and log_std for each action dimension
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if action_space is None:
            self.action_scale = torch.ones(self.action_dim).to(device)
            self.action_bias = torch.ones(self.action_dim).to(device)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2
            ).to(device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2
            ).to(device)

    def forward(self, obs):
        output = super().forward(obs)
        mean, log_std = torch.chunk(output, 2, dim=-1)

        # Constrain log_std to [-20, 10]
        log_std = torch.clamp(log_std, -20, 10)

        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # Sample using reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()

        # Squash using tanh
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # Calculate log probability, accounting for tanh squashing
        log_prob = normal.log_prob(x_t)
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
