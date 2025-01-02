import numpy as np
import numpy as np
from segment_tree import SumSegmentTree, MinSegmentTree


# class to store transitions
class ReplayMemory:
    def __init__(self, max_size=100000):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx, :] = np.asarray(
            transitions_new, dtype=object
        )
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        self.inds = np.random.choice(range(self.size), size=batch, replace=False)
        return self.transitions[self.inds, :]

    def get_all_transitions(self):
        return self.transitions[0 : self.size]


class PrioritizedExperienceReplay(ReplayMemory):
    def __init__(
        self,
        max_size=100000,
        beta_1=0.6,  # alpha in PER paper
        beta_2=0.4,  # beta in PER paper
        beta_increment=0.001,
        epsilon=1e-6,
        use_ere=False,
        eta_0=0.996,  # Initial ERE decay rate
        eta_T=1.0,  # Final ERE decay rate
        c_k_min=2500,  # Minimum ERE buffer size
    ):
        super().__init__(max_size)

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

        # ERE parameters
        self.use_ere = use_ere
        self.eta_0 = eta_0
        self.eta_T = eta_T
        self.c_k_min = c_k_min
        self.total_steps = 0
        self.c_k = max_size  # Start with full buffer

    def add_transition(self, transitions_new):
        """Add transition with maximum priority"""
        idx = self.current_idx
        super().add_transition(transitions_new)

        priority = self.max_priority**self.beta_1
        self.sum_tree[idx] = priority
        self.min_tree[idx] = priority

        if self.use_ere:
            self.total_steps += 1

    def _compute_c_k(self, total_steps):
        """Compute ERE c_k parameter using the paper's formula"""
        if self.total_steps == 0 or total_steps == 0:
            return self.size

        # Compute eta_t
        eta_t = self.eta_0 + (self.eta_T - self.eta_0) * (
            self.total_steps / total_steps
        )

        # Compute c_k using k/K ratio
        k = self.total_steps
        K = total_steps
        c_k = int(self.size * (eta_t ** ((k * 1000) / K)))

        # Ensure c_k stays within bounds
        c_k = max(self.c_k_min, min(c_k, self.size))
        return c_k

    def sample(self, batch=1, total_steps=None):
        """Sample transitions using priorities and ERE weights"""
        if batch > self.size:
            batch = self.size

        # Get effective buffer size for ERE
        if self.use_ere and total_steps is not None:
            effective_size = min(self.size, self._compute_c_k(total_steps))
        else:
            effective_size = self.size

        indices = []
        weights = np.zeros(batch, dtype=np.float32)
        total_p = self.sum_tree.sum(0, effective_size - 1)
        segment = total_p / batch

        # Sample from segments, considering only recent experiences
        for i in range(batch):
            a = segment * i
            b = segment * (i + 1)
            mass = np.random.uniform(a, b)
            idx = self.sum_tree.find_prefixsum_idx(mass)
            if idx >= effective_size:  # Ensure we only sample from recent experiences
                idx = np.random.randint(0, effective_size)
            indices.append(idx)

        indices = np.array(indices)

        # Calculate sampling weights
        p_min = self.min_tree.min() / total_p
        max_weight = (p_min * effective_size) ** (-self.beta_2)

        for i, idx in enumerate(indices):
            p_sample = self.sum_tree[idx] / total_p
            weight = (p_sample * effective_size) ** (-self.beta_2)
            weights[i] = weight / max_weight

        self.beta_2 = min(1.0, self.beta_2 + self.beta_increment)
        return self.transitions[indices], indices, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        priorities = (np.abs(td_errors) + self.epsilon) ** self.beta_1

        for idx, priority in zip(indices, priorities):
            self.sum_tree[idx] = priority
            self.min_tree[idx] = priority

        self.max_priority = max(self.max_priority, np.max(priorities))

    def get_statistics(self):
        """Return buffer statistics"""
        return {
            "size": self.size,
            "max_priority": self.max_priority,
            "beta_2": self.beta_2,
            "ere_enabled": self.use_ere,
            "ere_c_k": self.c_k if self.use_ere else None,
            "total_steps": self.total_steps,
        }
