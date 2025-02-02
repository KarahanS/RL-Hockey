import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        use_layer_norm=False,
        ln_eps=1e-5
    ):
        super(Critic, self).__init__()
        self.use_layer_norm = use_layer_norm

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        if self.use_layer_norm:
            self.ln1 = nn.LayerNorm(256, eps=ln_eps)

        self.l2 = nn.Linear(256, 256)
        if self.use_layer_norm:
            self.ln2 = nn.LayerNorm(256, eps=ln_eps)

        self.l3 = nn.Linear(256, 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        if self.use_layer_norm:
            self.ln4 = nn.LayerNorm(256, eps=ln_eps)

        self.l5 = nn.Linear(256, 256)
        if self.use_layer_norm:
            self.ln5 = nn.LayerNorm(256, eps=ln_eps)

        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        # state/action shape: (batch_size, state_dim/action_dim)
        sa = torch.cat([state, action], dim=1)

        # Q1 forward pass
        q1 = self.l1(sa)
        if self.use_layer_norm:
            q1 = self.ln1(q1)
        q1 = F.relu(q1)

        q1 = self.l2(q1)
        if self.use_layer_norm:
            q1 = self.ln2(q1)
        q1 = F.relu(q1)

        q1 = self.l3(q1)

        # Q2 forward pass
        q2 = self.l4(sa)
        if self.use_layer_norm:
            q2 = self.ln4(q2)
        q2 = F.relu(q2)

        q2 = self.l5(q2)
        if self.use_layer_norm:
            q2 = self.ln5(q2)
        q2 = F.relu(q2)

        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = self.l1(sa)
        if self.use_layer_norm:
            q1 = self.ln1(q1)
        q1 = F.relu(q1)

        q1 = self.l2(q1)
        if self.use_layer_norm:
            q1 = self.ln2(q1)
        q1 = F.relu(q1)

        q1 = self.l3(q1)
        return q1
