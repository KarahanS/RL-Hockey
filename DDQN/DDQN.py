import numpy as np

from DQN import DQNAgent, QFunction


class DDQNAgent(DQNAgent):
    def __init__(self, observation_space, action_space, tau=1e-3, **userconfig):
        super().__init__(observation_space, action_space, **userconfig)

        self.Q_target = QFunction(observation_dim=self._observation_space.shape[0], 
                                  action_dim=self._action_n,
                                  learning_rate = 0)
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
