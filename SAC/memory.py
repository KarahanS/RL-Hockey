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
        alpha=0.6,
        beta=0.4,
        beta_increment=0.001,
        epsilon=1e-6,
        use_ere=False,
        c_k=0.8,  # Scaling factor for ERE
        c_e=0.5,  # ERE exponent factor
    ):
        """
        Enhanced Prioritized Experience Replay Buffer with optional ERE support.

        Args:
            max_size: Maximum buffer size
            alpha: Prioritization exponent (0 = no prioritization, 1 = full)
            beta: Initial importance sampling weight
            beta_increment: Increment for beta annealing
            epsilon: Small constant for stability
            use_ere: Whether to use Emphasizing Recent Experience
            c_k: ERE scaling factor
            c_e: ERE exponent factor
        """
        super().__init__(max_size)

        # Initialize segment trees for efficient sampling
        tree_capacity = 1
        while tree_capacity < max_size:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

        # ERE parameters
        self.use_ere = use_ere
        self.c_k = c_k
        self.c_e = c_e
        self.episode_count = 0

    def _get_ere_weight(self, idx):
        """Calculate ERE weight based on recency"""
        if not self.use_ere:
            return 1.0

        # Normalize position to [0, 1]
        normalized_pos = idx / self.size
        # Calculate ERE weight
        ere_weight = (self.c_k * normalized_pos) ** self.c_e
        return ere_weight

    def add_transition(self, transitions_new):
        """Add transition with maximum priority"""
        idx = self.current_idx
        super().add_transition(transitions_new)

        priority = self.max_priority**self.alpha
        self.sum_tree[idx] = priority
        self.min_tree[idx] = priority

    def sample(self, batch=1):
        """Sample transitions using priorities and ERE weights"""
        if batch > self.size:
            batch = self.size

        indices = []
        weights = np.zeros(batch, dtype=np.float32)
        total_p = self.sum_tree.sum(0, self.size - 1)
        segment = total_p / batch

        # Sample from segments
        for i in range(batch):
            a = segment * i
            b = segment * (i + 1)
            mass = np.random.uniform(a, b)
            idx = self.sum_tree.find_prefixsum_idx(mass)
            indices.append(idx)

        indices = np.array(indices)

        # Calculate sampling weights
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.size) ** (-self.beta)

        for i, idx in enumerate(indices):
            p_sample = self.sum_tree[idx] / self.sum_tree.sum()
            weight = (p_sample * self.size) ** (-self.beta)
            weights[i] = weight / max_weight

            if self.use_ere:
                weights[i] *= self._get_ere_weight(idx)

        self.beta = min(1.0, self.beta + self.beta_increment)
        return self.transitions[indices], indices, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha

        for idx, priority in zip(indices, priorities):
            self.sum_tree[idx] = priority
            self.min_tree[idx] = priority

        self.max_priority = max(self.max_priority, np.max(priorities))

    def on_episode_end(self):
        """Called at the end of each episode for ERE updates"""
        if self.use_ere:
            self.episode_count += 1
            # Optionally adjust ERE parameters based on episode count
            self.c_k = max(0.4, self.c_k * 0.999)  # Decay scaling factor

    def get_statistics(self):
        """Return buffer statistics"""
        return {
            "size": self.size,
            "max_priority": self.max_priority,
            "beta": self.beta,
            "ere_enabled": self.use_ere,
            "ere_scaling": self.c_k if self.use_ere else None,
        }
