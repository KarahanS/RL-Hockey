import torch
import numpy as np


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, gpu=False):
        """gpu: Try to use GPU if available"""

        super(Feedforward, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        layer_sizes = [self.input_size] + self.hidden_sizes

        if torch.cuda.is_available() and gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.layers = torch.nn.ModuleList()
        for i, o in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(torch.nn.Linear(i, o, device=self.device, dtype=torch.float32))

        self.activations = [torch.nn.ReLU() for _ in self.layers]
        self.readout = torch.nn.Linear(
            self.hidden_sizes[-1], self.output_size, device=self.device, dtype=torch.float32
        )

    def forward(self, x):
        for layer, activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        return self.readout(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()
    
    def predict_gpu(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x)


class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[128, 128],
                 learning_rate=2e-4, gpu=False):
        super().__init__(input_size=observation_dim, hidden_sizes=hidden_sizes,
                         output_size=action_dim, gpu=gpu)
        self.optimizer=torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            eps=0.000001
        )
        self.loss = torch.nn.SmoothL1Loss()  # MSELoss()

    def fit(self, observations, actions, targets):
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass
        pred = self.Q_value(
            torch.from_numpy(observations).float(),
            torch.from_numpy(actions)
        )

        # Compute Loss
        loss = self.loss(pred, torch.from_numpy(targets).float())
        
        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def fit_gpu(self, observations, actions, targets):
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass
        pred = self.Q_value(observations, actions)
        # Compute Loss
        loss = self.loss(pred, targets)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def Q_value(self, observations: torch.Tensor, actions: torch.Tensor):
        return self.forward(observations).gather(1, actions[:, None])
    
    def maxQ(self, observations: np.ndarray) -> np.ndarray:
        """Return the highest Q value among all actions"""
        
        return np.max(self.predict(observations), axis=-1, keepdims=True)
    
    def maxQ_gpu(self, observations: torch.Tensor) -> torch.Tensor:
        """Return the highest Q value among all actions"""

        return torch.max(self.predict_gpu(observations), dim=-1, keepdim=True).values
    
    def argmaxQ(self, observations: np.ndarray) -> np.ndarray:
        """Return the action that maximizes the Q value"""

        return np.argmax(self.predict(observations), axis=-1, keepdims=True)
    
    def argmaxQ_gpu(self, observations: torch.Tensor) -> torch.Tensor:
        """Return the action that maximizes the Q value"""

        return torch.argmax(self.predict_gpu(observations), dim=-1, keepdim=True)

    def greedyAction(self, observations: np.ndarray) -> np.int64:
        return np.argmax(self.predict(observations), axis=-1)
    
    def greedyAction_gpu(self, observations: torch.Tensor) -> int:
        return torch.argmax(self.predict_gpu(observations), dim=-1).item()


class DuelingQFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[128, 128],
                 learning_rate=2e-4):
        super().__init__(
            input_size=observation_dim,
            hidden_sizes=hidden_sizes,
            output_size=action_dim,
            learning_rate=learning_rate
        )

        # TODO
