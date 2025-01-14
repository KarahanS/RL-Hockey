import os
import sys

from gymnasium import spaces
import numpy as np
import torch

root_dir = os.path.dirname(os.path.abspath("./"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from DDQN.q_function import QFunction
from DDQN.memory import Memory


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
            "eps": 0.05,            # Epsilon in epsilon greedy policies
            "hidden_sizes": [128, 128],
            "discount": 0.95,
            "buffer_size": int(1e5),
            "batch_size": 128,
            "learning_rate": 0.0002,
        }
        self._config.update(userconfig)
        self._eps = self._config['eps']
        
        self.buffer = Memory(max_size=self._config["buffer_size"])

        # Q Network
        self.Q = QFunction(
            observation_dim=self._observation_space.shape[0],
            hidden_sizes=self._config["hidden_sizes"],
            action_dim=self._action_n,
            learning_rate=self._config["learning_rate"],
        )

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        # epsilon greedy
        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
        else: 
            action = self._action_space.sample()
        return action
    
    def store_transition(self, transition):
        self.buffer.add_transition(transition)
            
    def train(self, iter_fit=32):
        losses = []

        for _ in range(iter_fit):
            # Sample from the replay buffer
            data = self.buffer.sample(batch=self._config['batch_size'])
            s = np.stack(data[:, 0])  # s_t
            a = np.stack(data[:, 1])  # a_t
            rew = np.stack(data[:, 2])[:, None]  # reward (batchsize, 1)
            s_prime = np.stack(data[:, 3])  # s_t+1
            done = np.stack(data[:, 4])[:, None]  # done signal (batchsize, 1)
            
            v_prime = self.Q.maxQ(s_prime)

            # Current state Q targets
            gamma=self._config['discount']
            td_target = rew + gamma * (1.0 - done) * v_prime
            
            # optimize the lsq objective
            fit_loss = self.Q.fit(s, a, td_target)
            
            losses.append(fit_loss)
        
        return losses


class DQNTargetAgent(DQNAgent):
    def __init__(self, observation_space, action_space, tau=1e-3, **userconfig):
        super().__init__(observation_space, action_space, **userconfig)

        self.Q_target = QFunction(
            observation_dim=self._observation_space.shape[0],
            hidden_sizes=self._config["hidden_sizes"],
            action_dim=self._action_n,
            learning_rate = 0
        )
        self._config["update_target_every"] = 20
        self._config.update(userconfig)

        self._tau = tau  # Polyak averaging parameter, 1 for hard update
        self._update_target_net(tau=1.0)  # Hard update at the beginning
        self.train_iter = 0
    
    def _update_target_net(self, tau):
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        self.Q_target.load_state_dict(self.Q.state_dict())
    
    def train(self, iter_fit=32):
        losses = []
        self.train_iter += 1
        if self.train_iter % self._config["update_target_every"] == 0:
            self._update_target_net(self._tau)
        for i in range(iter_fit):
            # Sample from the replay buffer
            data = self.buffer.sample(batch=self._config['batch_size'])

            s = np.stack(data[:,0])  # s_t
            a = np.stack(data[:,1])  # a_t
            rew = np.stack(data[:,2])[:,None]  # reward (batchsize,1)
            s_prime = np.stack(data[:,3])  # s_t+1
            done = np.stack(data[:,4])[:,None]  # done signal (batchsize,1)
            
            v_prime = self.Q_target.maxQ(s_prime)

            # Current state Q targets
            gamma = self._config['discount']                                                
            td_target = rew + gamma * (1.0-done) * v_prime
            
            # optimize the lsq objective
            fit_loss = self.Q.fit(s, a, td_target)
            
            losses.append(fit_loss)

        return losses

    def save_state(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        torch.save(
            self.Q.state_dict(),
            os.path.join(save_dir, "Q_model.ckpt")
        )

    def load_state(self, load_dir):
        self.Q.load_state_dict(
            torch.load(os.path.join(load_dir, "Q_model.ckpt"), weights_only=True)
        )
