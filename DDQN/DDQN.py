import os
import sys

import numpy as np
import torch

root_dir = os.path.dirname(os.path.abspath("./"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from DDQN.DQN import DQNTargetAgent
from DDQN.q_function import QFunction


class DDQNAgent(DQNTargetAgent):
    def __init__(self, observation_space, action_space, tau=1e-3, **userconfig):
        super().__init__(observation_space, action_space, tau=tau, **userconfig)

        # TODO

    def train(self, iter_fit=32):
        # TODO
        return super().train(iter_fit=iter_fit)
