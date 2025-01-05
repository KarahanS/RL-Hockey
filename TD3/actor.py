import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        use_layer_norm=False,
        ln_eps=1e-5
    ):
        super(Actor, self).__init__()
        self.use_layer_norm = use_layer_norm

        # First layer
        self.l1 = nn.Linear(state_dim, 256)
        if self.use_layer_norm:
            self.ln1 = nn.LayerNorm(256, eps=ln_eps)

        # Second layer
        self.l2 = nn.Linear(256, 256)
        if self.use_layer_norm:
            self.ln2 = nn.LayerNorm(256, eps=ln_eps)

        # Final layer
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        # state shape: (batch_size, state_dim)
        a = self.l1(state)
        if self.use_layer_norm:
            a = self.ln1(a)
        a = F.relu(a)

        a = self.l2(a)
        if self.use_layer_norm:
            a = self.ln2(a)
        a = F.relu(a)

        # Final layer
        a = self.l3(a)
        return self.max_action * torch.tanh(a)
