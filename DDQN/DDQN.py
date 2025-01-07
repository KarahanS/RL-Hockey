from DQN import DQNAgent


class DDQNAgent(DQNAgent):
    def __init__(self, observation_space, action_space, tau=1e-3, **userconfig):
        super().__init__(observation_space, action_space, **userconfig)
        self._tau = tau
        