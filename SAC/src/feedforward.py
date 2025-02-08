import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Adds current directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Adds parent directory
import torch
import torch.nn as nn
import numpy as np
from typing import List, Union, Type, Optional, Callable


class FeedForward(nn.Module):
    """
    A generic feedforward neural network that supports arbitrary architecture configurations.

    Parameters
    ----------
    input_size : int
        The input dimension of the network.
    hidden_sizes : List[int]
        List of hidden layer sizes. Each element specifies a layer's width.
    output_size : int
        The output dimension of the network.
    hidden_activation : Union[Type[nn.Module], Callable, List[Union[Type[nn.Module], Callable]]]
        Activation function(s) for hidden layers. Can be:
        - A single activation function used for all hidden layers
        - A list of activation functions, one per hidden layer
    output_activation : Optional[Union[Type[nn.Module], Callable]] = None
        Final activation function. If None, no activation is applied.
    layer_norm : bool = False
        Whether to apply layer normalization after each hidden layer.
    dropout_prob : float = 0.0
        Dropout probability between layers. If 0, no dropout is applied.
    init_method : str = 'default'
        Weight initialization method. Options: 'default', 'xavier', 'kaiming'
    device : torch.device = None
        Device to place the network on. If None, uses CUDA if available.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        hidden_activation: Union[
            Type[nn.Module], Callable, List[Union[Type[nn.Module], Callable]]
        ] = nn.Tanh,
        output_activation: Optional[Union[Type[nn.Module], Callable]] = None,
        layer_norm: bool = False,
        dropout_prob: float = 0.0,
        init_method: str = "default",
        device: torch.device = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        layer_sizes = (
            [input_size] + hidden_sizes + [output_size]
        )  # Build layer sizes including input and output

        if not isinstance(hidden_activation, list):
            hidden_activation = [hidden_activation] * len(hidden_sizes)
        assert len(hidden_activation) == len(
            hidden_sizes
        ), "Must provide activation for each hidden layer"

        layers = []
        for i in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[i], layer_sizes[i + 1])

            # Initialize weights
            if init_method == "xavier":
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
            elif init_method == "kaiming":
                nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
                nn.init.zeros_(linear.bias)

            layers.append(linear)

            if i < len(layer_sizes) - 2:  # For all except last layer
                if hidden_activation[i] is not None:
                    if isinstance(hidden_activation[i], type):
                        layers.append(hidden_activation[i]())
                    else:
                        layers.append(hidden_activation[i])

                if layer_norm:
                    layers.append(nn.LayerNorm(layer_sizes[i + 1]))

                if dropout_prob > 0:
                    layers.append(nn.Dropout(dropout_prob))

            # Add output activation if specified
            elif output_activation is not None and i == len(layer_sizes) - 2:
                if isinstance(output_activation, type):
                    layers.append(output_activation())
                else:
                    layers.append(output_activation)

        self.network = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        if x.device != self.device:
            x = x.to(self.device)
        return self.network(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions from numpy array input.

        Parameters
        ----------
        x : np.ndarray
            Input array

        Returns
        -------
        np.ndarray
            Predictions as numpy array
        """
        with torch.no_grad():
            x_tensor = torch.from_numpy(x.astype(np.float32))
            return self.forward(x_tensor).cpu().numpy()

    def get_parameter_count(self) -> dict:
        """
        Get the number of parameters in the network.

        Returns
        -------
        dict
            Dictionary containing total and trainable parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total_params, "trainable": trainable_params}
