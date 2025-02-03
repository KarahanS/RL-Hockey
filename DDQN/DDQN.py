import copy
import os
import sys

import numpy as np
import torch

root_dir = os.path.dirname(os.path.abspath("./"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from DDQN.DQN import TargetDQNAgent, DoubleDQNAgent
from DDQN.q_function import DuelingQFunction


class DuelingDQNBase:
    def ddqn_init(self, **userconfig):
        self._config["hidden_sizes_A"] = [128]
        self._config["hidden_sizes_V"] = [128]
        self._config.update(userconfig)

        self.Q = DuelingQFunction(
            observation_dim=self._observation_space.shape[0],
            hidden_sizes=self._config["hidden_sizes"],
            action_dim=self._action_n,
            learning_rate=self._config["learning_rate"],
            use_torch=self._config["use_torch"]
        )

        self.Q_target = copy.deepcopy(self.Q)


class DuelingDQNAgent(TargetDQNAgent, DuelingDQNBase):
    def __init__(self, observation_space, action_space, **userconfig):
        super().__init__(observation_space, action_space, **userconfig)
        self.ddqn_init(**userconfig)


class DoubleDuelingDQNAgent(DoubleDQNAgent, DuelingDQNBase):
    def __init__(self, observation_space, action_space, **userconfig):
        super().__init__(observation_space, action_space, **userconfig)
        self.ddqn_init(**userconfig)
