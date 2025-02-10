import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Adds current directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Adds parent directory
import numpy as np
from segment_tree import SumSegmentTree, MinSegmentTree


# ---------------------------
# Base ReplayMemory Class
# ---------------------------
class ReplayMemory:
    def __init__(self, max_size=100000):
        self.transitions = np.asarray([])  # Will be initialized on first transition
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)
        self.transitions[self.current_idx, :] = np.asarray(transitions_new, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        inds = np.random.choice(range(self.size), size=batch, replace=False)
        return self.transitions[inds, :]

    def get_all_transitions(self):
        return self.transitions[0 : self.size]
    
    def get_statitics(self):
        return {"size": self.size, "max_size": self.max_size}
    
    def get_state(self):
        return {
            "transitions": self.transitions,
            "size": self.size,
            "current_idx": self.current_idx,
            "max_size": self.max_size,
        }
    
    def set_state(self, state):
        try:
            self.transitions = state["transitions"]
            self.size = state["size"]
            self.current_idx = state["current_idx"]
            self.max_size = state["max_size"]
        except KeyError as e:
            print(f"Warning: Missing key in ReplayMemory restore: {e}")

    def clear(self):
        """Clears the memory and resets internal counters."""
        self.transitions = np.asarray([])  # Will be reinitialized on first add_transition call
        self.size = 0
        self.current_idx = 0

# ---------------------------
# Prioritized Experience Replay
# ---------------------------
class PrioritizedExperienceReplay(ReplayMemory):
    """Prioritized Experience Replay implementation using Segment Trees"""

    def __init__(
        self,
        max_size=100000,
        beta_1=0.6,  # alpha in PER paper
        beta_2=0.4,  # beta in PER paper
        beta_increment=0.001,
        epsilon=1e-6
    ):
        super().__init__(max_size)
        # Initialize segment trees with capacity as the next power of 2
        tree_capacity = 1
        while tree_capacity < max_size:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

    def add_transition(self, transitions_new):
        """Add transition with maximum priority"""
        idx = self.current_idx
        super().add_transition(transitions_new)
        priority = self.max_priority ** self.beta_1
        self.sum_tree[idx] = priority
        self.min_tree[idx] = priority

    def sample(self, batch=1, step=None, total_steps=None):
        if batch > self.size:
            batch = self.size

        indices = []
        weights = np.zeros(batch, dtype=np.float32)
        total_p = self.sum_tree.sum(0, self.size - 1)
        segment = total_p / batch

        # Sample from segments using priority
        for i in range(batch):
            a = segment * i
            b = segment * (i + 1)
            mass = np.random.uniform(a, b)
            idx = self.sum_tree.find_prefixsum_idx(mass)
            indices.append(idx)

        indices = np.array(indices)

        # Calculate importance sampling weights
        p_min = self.min_tree.min() / total_p
        max_weight = (p_min * self.size) ** (-self.beta_2)

        for i, idx in enumerate(indices):
            p_sample = self.sum_tree[idx] / total_p
            weight = (p_sample * self.size) ** (-self.beta_2)
            weights[i] = weight / max_weight

        self.beta_2 = min(1.0, self.beta_2 + self.beta_increment)
        transitions = self.transitions[indices]
        states = np.array([t[0] for t in transitions])
        actions = np.array([t[1] for t in transitions])
        rewards = np.array([t[2] for t in transitions])
        next_states = np.array([t[3] for t in transitions])
        dones = np.array([t[4] for t in transitions])

        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices, td_errors):
        priorities = (np.abs(td_errors) + self.epsilon) ** self.beta_1
        for idx, priority in zip(indices, priorities):
            self.sum_tree[idx] = priority
            self.min_tree[idx] = priority
        self.max_priority = max(self.max_priority, np.max(priorities))

    def get_statistics(self):
        return {
            "size": self.size,
            "max_priority": self.max_priority,
            "beta_2": self.beta_2,
        }
    
    def get_state(self):
        base_state = super().get_state()
        per_state = {
            "sum_tree": self.sum_tree.get_state(),  # using the method
            "min_tree": self.min_tree.get_state(),  # using the method
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "beta_increment": self.beta_increment,
            "epsilon": self.epsilon,
            "max_priority": self.max_priority,
        }
        base_state.update(per_state)
        return base_state

    def set_state(self, state):
        super().set_state(state)
        
        sum_tree_state = state.get("sum_tree")
        if sum_tree_state is not None:
            self.sum_tree.set_state(sum_tree_state)
        else:
            print("Warning: 'sum_tree' key missing. Using current sum_tree state.")
            
        min_tree_state = state.get("min_tree")
        if min_tree_state is not None:
            self.min_tree.set_state(min_tree_state)
        else:
            print("Warning: 'min_tree' key missing. Using current min_tree state.")
            
        self.beta_1 = state.get("beta_1", self.beta_1)
        self.beta_2 = state.get("beta_2", self.beta_2)
        self.beta_increment = state.get("beta_increment", self.beta_increment)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.max_priority = state.get("max_priority", self.max_priority)

# ERE Prioritized Experience Replay
# ---------------------------
class EREPrioritizedExperienceReplay(PrioritizedExperienceReplay):
    """Prioritized Experience Replay with Emphasized Recent Experience"""

    def __init__(
        self,
        max_size=100000,
        beta_1=0.6,
        beta_2=0.4,
        beta_increment=0.001,
        epsilon=1e-6,
        eta_0=0.996,
        eta_T=1.0,
        c_k_min=2500,
    ):
        super().__init__(max_size, beta_1, beta_2, beta_increment, epsilon)
        # ERE parameters
        self.eta_0 = eta_0
        self.eta_T = eta_T
        self.c_k_min = c_k_min

    def _compute_c_k(self, step, total_steps):
        if total_steps == 0:
            return self.size
        k = step
        K = total_steps
        eta_t = self.eta_0 + (self.eta_T - self.eta_0) * (step / total_steps)
        c_k = int(self.size * (eta_t ** ((k * 1000) / K)))
        c_k = max(self.c_k_min, min(c_k, self.size))
        return c_k

    def _get_recent_indices(self, c_k):
        if self.size < c_k:
            return range(self.size)
        start_idx = (self.current_idx - c_k) % self.size
        return [(start_idx + i) % self.size for i in range(c_k)]

    def sample(self, batch=1, step=None, total_steps=None):
        if batch > self.size:
            batch = self.size
        c_k = self._compute_c_k(step, total_steps)
        valid_indices = self._get_recent_indices(c_k)
        indices = []
        weights = np.zeros(batch, dtype=np.float32)
        total_p = sum(self.sum_tree[idx] for idx in valid_indices)
        segment = total_p / batch
        for i in range(batch):
            a = segment * i
            b = segment * (i + 1)
            mass = np.random.uniform(a, b)
            cumsum = 0
            for idx in valid_indices:
                cumsum += self.sum_tree[idx]
                if cumsum > mass:
                    indices.append(idx)
                    break
            if len(indices) <= i:
                indices.append(np.random.choice(valid_indices))
        indices = np.array(indices)
        p_min = min(self.sum_tree[idx] for idx in valid_indices) / total_p
        max_weight = (p_min * len(valid_indices)) ** (-self.beta_2)
        for i, idx in enumerate(indices):
            p_sample = self.sum_tree[idx] / total_p
            weight = (p_sample * len(valid_indices)) ** (-self.beta_2)
            weights[i] = weight / max_weight
        self.beta_2 = min(1.0, self.beta_2 + self.beta_increment)
        transitions = self.transitions[indices]
        states = np.array([t[0] for t in transitions])
        actions = np.array([t[1] for t in transitions])
        rewards = np.array([t[2] for t in transitions])
        next_states = np.array([t[3] for t in transitions])
        dones = np.array([t[4] for t in transitions])
        return (states, actions, rewards, next_states, dones), indices, weights

    def get_statistics(self):
        stats = super().get_statistics()
        stats.update(
            {
                "ere_eta0": self.eta_0,
                "ere_etaT": self.eta_T,
                "ere_c_k_min": self.c_k_min,
            }
        )
        return stats
    
    # New: Full state for ERE PER (includes PER state and ERE parameters)
    def get_state(self):
        base_state = super().get_state()
        ere_state = {
            "ere_eta0": self.eta_0,
            "ere_etaT": self.eta_T,
            "ere_c_k_min": self.c_k_min,
        }
        base_state.update(ere_state)
        return base_state
    
    # New: Restore full state for ERE PER
    def set_state(self, state):
        super().set_state(state)
        self.eta_0 = state.get("ere_eta0", self.eta_0)
        self.eta_T = state.get("ere_etaT", self.eta_T)
        self.c_k_min = state.get("ere_c_k_min", self.c_k_min)
