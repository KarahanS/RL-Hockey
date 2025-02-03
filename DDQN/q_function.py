import torch
import numpy as np


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, use_torch=True):
        """use_torch: Use PyTorch functions, try to use GPU if available"""

        super(Feedforward, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        layer_sizes = [self.input_size] + self.hidden_sizes

        if torch.cuda.is_available() and use_torch:
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
            return self.forward(
                torch.from_numpy(x.astype(np.float32)).to(self.device)
            ).cpu().numpy()
    
    def predict_torch(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x)


class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[128, 128],
                 learning_rate=2e-4, use_torch=True):
        super().__init__(input_size=observation_dim, hidden_sizes=hidden_sizes,
                         output_size=action_dim, use_torch=use_torch)
        self.optimizer=torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            eps=0.000001
        )
        self.loss = torch.nn.SmoothL1Loss()  # MSELoss()

    # FIXME: CPU training functions will give device errors since Q can be on GPU

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

    def fit_torch(self, observations, actions, targets):
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
    
    def maxQ_torch(self, observations: torch.Tensor) -> torch.Tensor:
        """Return the highest Q value among all actions"""

        return torch.max(self.predict_torch(observations), dim=-1, keepdim=True).values
    
    def argmaxQ(self, observations: np.ndarray) -> np.ndarray:
        """Return the action that maximizes the Q value"""

        return np.argmax(self.predict(observations), axis=-1, keepdims=True)
    
    def argmaxQ_torch(self, observations: torch.Tensor) -> torch.Tensor:
        """Return the action that maximizes the Q value"""

        return torch.argmax(self.predict_torch(observations), dim=-1, keepdim=True)

    def greedyAction(self, observations: np.ndarray) -> np.int64:
        return np.argmax(self.predict(observations), axis=-1)
    
    def greedyAction_torch(self, observations: torch.Tensor) -> int:
        return torch.argmax(self.predict_torch(observations), dim=-1).item()


class DuelingFeedforward(Feedforward):
    def __init__(self, input_size, hidden_sizes, hidden_sizes_A, hidden_sizes_V, output_size,
                 use_torch=True):
        """use_torch: Use PyTorch functions, try to use GPU if available"""

        Feedforward.__init__(
            self,
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            use_torch=use_torch,
        )

        self.hidden_sizes_A = hidden_sizes_A
        self.hidden_sizes_V = hidden_sizes_V

        if torch.cuda.is_available() and use_torch:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        layer_sizes_A = [self.hidden_sizes[-1]] + hidden_sizes_A
        layer_sizes_V = [self.hidden_sizes[-1]] + hidden_sizes_V

        self.layers_A = torch.nn.ModuleList()
        for i, o in zip(layer_sizes_A[:-1], layer_sizes_A[1:]):
            self.layers_A.append(torch.nn.Linear(i, o, device=self.device, dtype=torch.float32))
        self.activations_A = [torch.nn.ReLU() for _ in self.layers_A]
        self.readout_A = torch.nn.Linear(
            self.hidden_sizes_A[-1], self.output_size, device=self.device, dtype=torch.float32
        )
            
        self.layers_V = torch.nn.ModuleList()
        for i, o in zip(layer_sizes_V[:-1], layer_sizes_V[1:]):
            self.layers_V.append(torch.nn.Linear(i, o, device=self.device, dtype=torch.float32))
        self.activations_V = [torch.nn.ReLU() for _ in self.layers_V]
        self.readout_V = torch.nn.Linear(
            self.hidden_sizes_V[-1], 1, device=self.device, dtype=torch.float32
        )
        
    def forward(self, x):
        for layer, activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        # Note: no readout layer here

        x_A = x
        for layer, activation_fun in zip(self.layers_A, self.activations_A):
            x_A = activation_fun(layer(x_A))
        x_A = self.readout_A(x_A)

        x_V = x
        for layer, activation_fun in zip(self.layers_V, self.activations_V):
            x_V = activation_fun(layer(x_V))
        x_V = self.readout_V(x_V)

        return x_V + x_A - x_A.mean(dim=-1, keepdim=True)


class DuelingQFunction(DuelingFeedforward, QFunction):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[128, 128], hidden_sizes_A=[128],
                 hidden_sizes_V=[128], learning_rate=2e-4, use_torch=True):
        DuelingFeedforward.__init__(
            self,
            input_size=observation_dim,
            hidden_sizes=hidden_sizes,
            hidden_sizes_A=hidden_sizes_A,
            hidden_sizes_V=hidden_sizes_V,
            output_size=action_dim,
            use_torch=use_torch,
        )

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            eps=0.000001
        )

        self.loss = torch.nn.SmoothL1Loss()  # MSELoss()
    
    # fit, Q_value, maxQ, argmaxQ, greedyAction, and their torch versions
    #   are inherited from QFunction
