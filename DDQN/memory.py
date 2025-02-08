import os
import sys

import numpy as np
import torch

root_dir = os.path.dirname(os.path.abspath("./"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from SAC.src.segment_tree import SegmentTree, MinSegmentTree, SumSegmentTree


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
        self._current_idx = 0
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
        self.transitions_states[self._current_idx] = states

        # Store other fields (scalar values)
        self.transitions_actions[self._current_idx] = act
        self.transitions_rewards[self._current_idx] = rew
        self.transitions_dones[self._current_idx] = done

        self.size = min(self.size + 1, self.max_size)
        self._current_idx = (self._current_idx + 1) % self.max_size

    def sample(self, batch_size=1):
        if batch_size > self.size:
            batch_size = self.size
        
        #self.inds=np.random.choice(range(self.size), size=batch, replace=False)
        self.inds = torch.randperm(self.size)[0:batch_size]  # random sampling WITHOUT replacement

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
    

class MemoryPERTorch(MemoryTorch):
    def __init__(self, max_size: int = 1e5, alpha: int = 0.6, beta: int = 0.4, epsilon: int = 1e-6,
                 device: torch.device = torch.device('cpu')):
        super().__init__(max_size, device)
        # TODO: check if memorytorch is compatible, and implement this class
        # TODO: beta increment and max?
        self._alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self._max_priority = 1.0

        segtree_capacity = 1
        while segtree_capacity < max_size:
            segtree_capacity *= 2
        self._segtree_sum = SumSegmentTree(segtree_capacity)
        self._segtree_min = MinSegmentTree(segtree_capacity)
    
    def add_transition(self, transitions_new: tuple):
        idx = self._current_idx
        MemoryTorch.add_transition(self, transitions_new)
        self._segtree_sum[idx] = self._max_priority ** self._alpha
        self._segtree_min[idx] = self._max_priority ** self._alpha
    
    def _sample_proportional(self, batch_size: int):
        p_total = self._segtree_sum.sum(0, self.size - 1)
        segment = p_total / batch_size

        batch_inds = []
        for i in range(batch_size):
            mass = segment * (np.random.rand() + i)
            idx = self._segtree_sum.find_prefixsum_idx(mass)
            batch_inds.append(idx)
        
        return batch_inds
    
    def sample(self, batch_size=1):
        if batch_size > self.size:
            batch_size = self.size
        
        batch_inds = self._sample_proportional(batch_size)
        #self.inds = torch.tensor(batch_inds, device=self.device)
        weights = []

        p_min = self._segtree_min.min() / self._segtree_sum.sum()
        max_weight = (p_min * self.size) ** (-self.beta)

        for idx in batch_inds:
            p_sample = self._segtree_sum[idx] / self._segtree_sum.sum()
            weight = (p_sample * self.size) ** (-self.beta)
            weight = weight / max_weight
            weights.append(weight)
        
        states = self.transitions_states[batch_inds]
        ob = states[:, 0, :]
        ob_next = states[:, 1, :]

        act = self.transitions_actions[batch_inds]
        rew = self.transitions_rewards[batch_inds]
        done = self.transitions_dones[batch_inds]

        # Convert to column vector to avoid broadcasting
        rew = rew[:, None]
        done = done[:, None]
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        inds = torch.tensor(batch_inds, device=self.device, dtype=torch.int64)

        return (ob, act, rew, ob_next, done, weights, inds)
        

    def update_priorities(self, inds: torch.Tensor, priorities: torch.Tensor):
        for idx, priority in zip(inds, priorities):
            assert priority > 0
            assert 0 <= idx < self.size

            self._segtree_sum[idx] = priority ** self._alpha
            self._segtree_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)
