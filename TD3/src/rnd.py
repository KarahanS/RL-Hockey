# rnd.py
import torch
import torch.nn as nn
import torch.optim as optim

class RNDNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(RNDNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class RND:
    def __init__(self, state_dim, device, hidden_dim=128, lr=1e-4):
        self.device = device
        self.target_network = RNDNetwork(state_dim, hidden_dim).to(self.device)
        self.predictor_network = RNDNetwork(state_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.predictor_network.parameters(), lr=lr)
        # Freeze target network
        for param in self.target_network.parameters():
            param.requires_grad = False
        self.loss_fn = nn.MSELoss(reduction='none')  # Use 'none' for element-wise loss
    
    def compute_intrinsic_reward(self, states):
        """
        Computes intrinsic rewards for a batch of states.
        Args:
            states (torch.Tensor): Tensor of states with shape [batch_size, state_dim]
        Returns:
            torch.Tensor: Intrinsic rewards with shape [batch_size, 1]
        """
        with torch.no_grad():
            target_output = self.target_network(states)
        predictor_output = self.predictor_network(states)
        # Compute MSE loss per sample
        intrinsic_reward = self.loss_fn(predictor_output, target_output).mean(dim=1, keepdim=True)
        return intrinsic_reward  # Shape: [batch_size, 1]
    
    def update_predictor(self, states):
        """
        Updates the predictor network using a batch of states.
        Args:
            states (torch.Tensor): Tensor of states with shape [batch_size, state_dim]
        Returns:
            torch.Tensor: Mean loss over the batch
        """
        target_output = self.target_network(states).detach()
        predictor_output = self.predictor_network(states)
        loss = self.loss_fn(predictor_output, target_output).mean(dim=1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
