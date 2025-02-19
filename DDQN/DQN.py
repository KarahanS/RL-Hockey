import copy
import os
import sys

from gymnasium import spaces
import numpy as np
import torch

root_dir = os.path.dirname(os.path.abspath("./"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from DDQN.dqn_q_function import QFunction
from DDQN.dqn_memory import Memory, MemoryTorch, MemoryPERTorch


class DQNAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """
    def __init__(self, observation_space, action_space, **userconfig):
        if not isinstance(observation_space, spaces.box.Box):
            raise ValueError('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.discrete.Discrete):
            raise ValueError('Action space {} incompatible with {}.' \
                                   ' (Reqire Discrete.)'.format(action_space, self))

        self._observation_space = observation_space
        self._action_space = action_space
        self._action_n = action_space.n
        self._config = {
            "epsilon": 0.2,  # Epsilon in epsilon greedy policies
            "epsilon_decay_rate": 0.999,
            "epsilon_min": 0.1,
            "hidden_sizes": [128, 128],
            "discount": 0.95,
            "buffer_size": int(1e5),
            "batch_size": 128,
            "learning_rate": 0.0002,
        }
        self._config.update(userconfig)
        self.eps = self._config['epsilon']
        self._eps_decay_rate = self._config['epsilon_decay_rate']
        self._eps_min = self._config['epsilon_min']
        self._batch_size = self._config['batch_size']
        
        self._per = self._config.get("per", False)
        self._use_torch = self._config.get("use_torch", False)
        
        # Q Network
        self.Q = QFunction(
            observation_dim=self._observation_space.shape[0],
            hidden_sizes=self._config["hidden_sizes"],
            action_dim=self._action_n,
            learning_rate=self._config["learning_rate"],
            use_torch=self._use_torch
        )

        self.train_device = self.Q.device
        if self._use_torch:
            if self._per:
                self.buffer = MemoryPERTorch(max_size=self._config["buffer_size"], device=self.train_device)
            else:
                self.buffer = MemoryTorch(max_size=self._config["buffer_size"], device=self.train_device)
        else:
            if self._per:
                raise NotImplementedError("PER is not implemented for numpy version")
            self.buffer = Memory(max_size=self._config["buffer_size"])

    # TODO: If the server allows so, deprecate numpy-only alternatives and rename the torch
    #   versions to the original names

    def act(self, observation: np.ndarray, explore=False):
        """explore: Allow action exploration. Should not use in evaluation"""
        
        # Epsilon greedy
        if (not explore) or np.random.random() > self.eps:
            # Greedy action
            action = self.Q.greedyAction(observation)
        else:
            # Random action
            action = self._action_space.sample()
        
        return action
    
    def act_torch(self, observation: torch.Tensor, explore=False):
        """explore: Allow action exploration. Should not use in evaluation"""
        
        # Epsilon greedy
        if (not explore) or np.random.random() > self.eps:
            # Greedy action
            action = self.Q.greedyAction_torch(observation)
        else:
            # Random action
            action = self._action_space.sample()
        
        return action
    
    def store_transition(self, transition: tuple):
        self.buffer.add_transition(transition)

    def _training_objective(self, sampled_data):
        """Return sampled states, actions, and the value to minimize in the Q-learning update"""

        s = np.stack(sampled_data[:, 0])  # s_t
        a = np.stack(sampled_data[:, 1])  # a_t
        rew = np.stack(sampled_data[:, 2])  # reward
        s_prime = np.stack(sampled_data[:, 3])  # s_t+1
        done = np.stack(sampled_data[:, 4])  # done signal
        
        # Convert to column vector to avoid broadcasting
        rew = rew[:, None]
        done = done[:, None]

        v_prime = self.Q.maxQ(s_prime)

        # Current state Q targets
        gamma=self._config['discount']
        td_target = rew + gamma * (1.0 - done) * v_prime

        return s, a, td_target

    def _training_objective_torch(self, sampled_data):
        """Return sampled states, actions, and the value to minimize in the Q-learning update"""

        s, a, rew, s_prime, done = sampled_data

        v_prime = self.Q.maxQ_torch(s_prime)

        # Current state Q targets
        gamma = self._config['discount']
        td_target = rew + gamma * (~done) * v_prime  # FIXME: Shape is [128, 128] instead of correct [128, 1] as in numpy version

        return s, a, td_target

    def train(self, iter_fit=32):
        if self._per:
            raise NotImplementedError("PER is not implemented for numpy version")

        losses = []
        for _ in range(iter_fit):
            # Sample from the replay buffer
            data = self.buffer.sample(batch_size=self._batch_size)
            s, a, td_target = self._training_objective(data)

            # Optimize the lsq objective
            fit_loss = self.Q.fit(s, a, td_target)
            losses.append(fit_loss)
        
        # Decay epsilon
        self.eps = max(self._eps_min, self.eps * self._eps_decay_rate)

        return losses

    def train_torch(self, iter_fit=32):
        losses = []

        for _ in range(iter_fit):
            # Sample from the replay buffer
            data = self.buffer.sample(batch_size=self._batch_size)
            s, a, td_target = self._training_objective_torch(data[:5])
            
            if self._per:
                loss_weights = data[5]
                indices = data[6]
                q_value = self.Q.Q_value(s, a)
            else:
                loss_weights = None

            # Optimize the lsq objective
            fit_loss = self.Q.fit_torch(s, a, td_target, loss_weights)
            losses.append(fit_loss)

            if self._per:
                #priorities = torch.abs(fit_loss.flatten()) + self.buffer.epsilon
                priorities = torch.abs(td_target - q_value) + self.buffer.epsilon
                self.buffer.update_priorities(indices, priorities.flatten())
        
        # Decay epsilon
        self.eps = max(self._eps_min, self.eps * self._eps_decay_rate)

        return losses

    def save_state(self, save_path):
        save_dir = os.path.dirname(save_path)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        torch.save(self.Q.state_dict(), save_path)

    def load_state(self, load_path):  # TODO: merge args
        self.Q.load_state_dict(
            #torch.load(os.path.join(load_dir, filename), weights_only=True)
            torch.load(load_path, weights_only=True)
        )


class TargetDQNAgent(DQNAgent):
    def __init__(self, observation_space, action_space, **userconfig):
        super().__init__(observation_space, action_space, **userconfig)

        self._config["tau"] = 1e-3
        self._config["update_target_every"] = 20
        self._config.update(userconfig)

        self.Q_target = copy.deepcopy(self.Q)

        self.train_iter = 0

        self._tau = self._config["tau"]  # Polyak averaging parameter, 1 for hard update
        # Hard update at the beginning
        if self._use_torch:
            self._update_target_net_torch(tau=1.0)
        else:
            self._update_target_net(tau=1.0)
        

    def _update_target_net(self, tau):
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        self.Q_target.load_state_dict(self.Q.state_dict())
    
    def _update_target_net_torch(self, tau):
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        self.Q_target.load_state_dict(self.Q.state_dict())
    
    def _training_objective(self, sampled_data):
        """Return sampled states, actions, and the value to minimize in the Q-learning update"""

        s = np.stack(sampled_data[:, 0])  # s_t
        a = np.stack(sampled_data[:, 1])  # a_t
        rew = np.stack(sampled_data[:, 2])  # reward
        s_prime = np.stack(sampled_data[:, 3])  # s_t+1
        done = np.stack(sampled_data[:, 4])  # done signal

        # Convert to column vector to avoid broadcasting
        rew = rew[:, None]
        done = done[:, None]

        v_prime = self.Q_target.maxQ(s_prime)

        # Current state Q targets
        gamma = self._config['discount']
        td_target = rew + gamma * (1.0 - done) * v_prime

        return s, a, td_target

    def _training_objective_torch(self, sampled_data):
        """Return sampled states, actions, and the value to minimize in the Q-learning update"""

        s, a, rew, s_prime, done = sampled_data
        v_prime = self.Q_target.maxQ_torch(s_prime)

        # Current state Q targets
        gamma = self._config['discount']
        td_target = rew + gamma * (~done) * v_prime

        return s, a, td_target

    def train(self, iter_fit=32):
        if self._per:
            raise NotImplementedError("PER is not implemented for numpy version")

        losses = []
        self.train_iter += 1
        if self.train_iter % self._config["update_target_every"] == 0:
            self._update_target_net(self._tau)

        for _ in range(iter_fit):
            # Sample from the replay buffer
            data = self.buffer.sample(batch_size=self._batch_size)
            s, a, td_objective = self._training_objective(data)
            
            # Optimize the lsq objective
            fit_loss = self.Q.fit(s, a, td_objective)
            losses.append(fit_loss)

        # Decay epsilon
        self.eps = max(self._eps_min, self.eps * self._eps_decay_rate)

        return losses

    def train_torch(self, iter_fit=32):
        losses = []
        self.train_iter += 1
        if self.train_iter % self._config["update_target_every"] == 0:
            self._update_target_net_torch(self._tau)

        for _ in range(iter_fit):
            # Sample from the replay buffer
            data = self.buffer.sample(batch_size=self._batch_size)
            s, a, td_objective = self._training_objective_torch(data[:5])

            if self._per:
                loss_weights = data[5]
                indices = data[6]
                q_value = self.Q.Q_value(s, a)
            else:
                loss_weights = None
            
            # Optimize the lsq objective
            fit_loss = self.Q.fit_torch(s, a, td_objective, loss_weights)
            losses.append(fit_loss)

            if self._per:
                #priorities = torch.abs(fit_loss.flatten()) + self.buffer.epsilon
                priorities = torch.abs(td_objective - q_value) + self.buffer.epsilon
                self.buffer.update_priorities(indices, priorities.flatten())
        
        # Decay epsilon
        self.eps = max(self._eps_min, self.eps * self._eps_decay_rate)

        return losses


class DoubleDQNAgent(TargetDQNAgent):
    def __init__(self, observation_space, action_space, **userconfig):
        super().__init__(observation_space, action_space, **userconfig)

    def _training_objective(self, sampled_data):
        """Return sampled states, actions, and the value to minimize in the Q-learning update"""

        s = np.stack(sampled_data[:, 0])  # s_t
        a = np.stack(sampled_data[:, 1])  # a_t
        rew = np.stack(sampled_data[:, 2])  # reward
        s_prime = np.stack(sampled_data[:, 3])  # s_t+1
        done = np.stack(sampled_data[:, 4])  # done signal

        # Convert to column vector to avoid broadcasting
        rew = rew[:, None]
        done = done[:, None]

        v_prime = self.Q_target.Q_value(
            observations=torch.from_numpy(s_prime).float(),
            actions=torch.from_numpy(self.Q.argmaxQ(s_prime)).long().flatten()
        ).detach().numpy()

        # Current state Q targets
        gamma = self._config['discount']
        td_target = rew + gamma * (1.0 - done) * v_prime

        return s, a, td_target

    def _training_objective_torch(self, sampled_data):
        """Return sampled states, actions, and the value to minimize in the Q-learning update"""

        s, a, rew, s_prime, done = sampled_data

        v_prime = self.Q_target.Q_value(
            observations=s_prime,
            actions=self.Q.argmaxQ_torch(s_prime).long().flatten()
        )

        # Current state Q targets
        gamma = self._config['discount']
        td_target = rew + gamma * (~done) * v_prime

        return s, a, td_target
