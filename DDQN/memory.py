import numpy as np
import torch


# class to store transitions
class Memory():
    def __init__(self, max_size=1e5):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size=max_size

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx,:] = np.asarray(transitions_new, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        self.inds=np.random.choice(range(self.size), size=batch, replace=False)
        return self.transitions[self.inds,:]

    def get_all_transitions(self):
        return self.transitions[0:self.size]


class MemoryTorch():
    def __init__(self, max_size: int = 1e5, device: torch.device = torch.device('cpu')):
        self.size = 0
        self.current_idx = 0
        self.max_size=max_size
        self.device = device

        # Current and next states of transitions:
        # To be initialized in add_transition according to observation vector dimensions
        self.transitions_states = None  
        # Other fields:
        self.transitions_actions = torch.empty(
            (self.max_size,), dtype=torch.int64, device=self.device  # Need int64 for indexing
        )
        self.transitions_rewards = torch.empty(
            (self.max_size,), dtype=torch.float32, device=self.device
        )
        self.transitions_dones = torch.empty(
            (self.max_size,), dtype=torch.bool, device=self.device
        )

    def add_transition(self, transitions_new: tuple):
        ob, act, rew, ob_next, done = transitions_new

        if self.transitions_states is None:
            # First transition: Initialize states tensor with correct dimensions
            self.transitions_states = torch.empty(
                (self.max_size, 2, ob.shape[0]), dtype=torch.float32, device=self.device
            )

        # Store current and next states
        ob = torch.Tensor(ob)
        ob_next = torch.Tensor(ob_next)
        states = torch.stack((ob, ob_next), dim=0).to(self.device)
        self.transitions_states[self.current_idx] = states

        # Store other fields (scalar values)
        self.transitions_actions[self.current_idx] = act
        self.transitions_rewards[self.current_idx] = rew
        self.transitions_dones[self.current_idx] = done

        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        
        #self.inds=np.random.choice(range(self.size), size=batch, replace=False)
        self.inds = torch.randperm(self.size)[0:batch]  # random sampling WITHOUT replacement

        states = self.transitions_states[self.inds]
        ob = states[:, 0, :]
        ob_next = states[:, 1, :]

        act = self.transitions_actions[self.inds]
        rew = self.transitions_rewards[self.inds]
        done = self.transitions_dones[self.inds]
        
        # Convert to column vector to avoid broadcasting
        rew = rew[:, None]
        done = done[:, None]

        return (ob, act, rew, ob_next, done)
