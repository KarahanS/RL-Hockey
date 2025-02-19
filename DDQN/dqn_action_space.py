import numpy as np
from gymnasium.spaces.discrete import Discrete


class CustomActionSpace(Discrete):
    # [axial_movement_x, axial_movement_y, rotation, shooting]
        # -1: left, down, counter-clockwise
        # 0: no action
        # 1: right, up, clockwise
    actions = [
            # Axial movement
            [0, 0, 0, 0],
            [-1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 1, 0, 0],

            # Rotation movement
            [0, 0, -1, 0],
            [0, 0, 1, 0],

            # Diagonal movement
            [-1, -1, 0, 0],
            [-1, 1, 0, 0],
            [1, -1, 0, 0],
            [1, 1, 0, 0],

            # Diagonal + rotation movement
            [-1, -1, -1, 0],
            [-1, -1, 1, 0],
            [-1, 1, -1, 0],
            [-1, 1, 1, 0],
            [1, -1, -1, 0],
            [1, -1, 1, 0],
            [1, 1, -1, 0],
            [1, 1, 1, 0],

            # Axial + rotation movement
            # Omit for simplicity

            # Shooting
            [0, 0, 0, 1]
            # Omit non-standing shooting for simplicity
    ]
    
    def __init__(self):
        self.n = len(self.actions)

    def sample(self):
        return np.random.randint(self.n)

    @staticmethod
    def discrete_to_continuous(action: int):
        return np.array(CustomActionSpace.actions[action])

    def __len__(self):
        return self.n
