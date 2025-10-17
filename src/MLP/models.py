from dataclasses import dataclass

from src.MLP.Dense import Dense
from src.MLP.loss_functions import BCE
from src.MLP.optimizer import Optimizer


@dataclass
class DenseConfig:
    """
    Data model containing configurations to create a Dense layer
    """
    input_shape: int
    output_shape: int
    activation: str = "sigmoid"
    weights_init: str | None = None


class MLP:
    def __init__(
            self,
            network_config: list[DenseConfig],
            epoch: int = 1000,
            lr: float = 0.005,
            es_threshold: float = 0.0001,
            batch_size: int = 16,
            print_freq: int = 100,
        ) -> None:
        """
        Initialize the Simple Neural Network with specified layers.

        Args:
            input_shape (tuple): A tuple of input examples, input features.
            output_shape (integer): Number of output classifications.
            out_activation: activation function for output layer
        """
        if len(network_config) < 3:
            raise ValueError("network_config must contain at least 3 layers (or two hidden layers)")

        # Build a dynamic list of layers from the provided configs
        self.layers: list[Dense] = []
        for cfg in network_config:
            self.layers.append(
                Dense(cfg.input_shape, cfg.output_shape, activation=cfg.activation)
            )

        # cache of activations (post-activation outputs) per layer from last forward pass
        self._activations: list = []
        self.loss_function = BCE() # use Binary cross Entropy to calculate Loss
        self.optimizer = Optimizer(lr)

    def forward(self, x):
        """
        Perform the forward pass through the entire network.

        Args:
            x: Input data of shape (m, input_size), where m is the number of samples.

        Return:
            The output of the network after the forward pass.
        """
        # Store intermediate activation values
        a = x
        self._activations = []
        for layer in self.layers:
            a = layer.forward(a)
            self._activations.append(a)
        return a

    def backward(self, y_true, y_pred):
        """
        Perform the backward pass through the entire network.

        Args:
            y_true: Ground truth
            y_pred: Predicted values by the model (i.e. y_hat)
        """
        # Upstream gradient from loss w.r.t. network output (dL/dA_L)
        dA = self.loss_function.loss_derivative(y_true, y_pred)

        # Backpropagate through layers in reverse order
        # Convert dA to dZ for the last layer using its activation derivative
        last_idx = len(self.layers) - 1
        dz = dA * self.layers[last_idx].activation.backward(self._activations[last_idx])

        # note: if any of the following case, the dL/dz3 is a special case of y_pred - y_true
        # . 1. Loss function == Classical Cross Entropy, Output Activation == Softmax
        # . 2. Loss function == Binary Cross Entropy, Output Acitvation == Sigmoid

        # Hidden layers
        d_prev = self.layers[last_idx].backward(dz)  # dL/dA_prev
        for i in range(len(self.layers) - 2, -1, -1):
            dz = d_prev * self.layers[i].activation.backward(self._activations[i])
            d_prev = self.layers[i].backward(dz)

    def update(self, optimizer: Optimizer):
        """
        Update model parameter

        Args:
            optimizer: An instance of Optimizer Class
        """
        for layer in self.layers:
            optimizer.sgd(layer)


# if __name__ == "__main__":
#     # Example:
#     input_size = 10 # Example feature size
#     input_data = np.random.rand(1, input_size)
#     output_data = np.ones((1, 1))  # Fix shape to match model output
#     model = MLP(input_data.shape)

#     # Forward pass example with random input
#     output = model.forward(input_data)
#     print("Model output:", output)
#     print("Target output:", output_data)

#     # Backward pass
#     model.backward(output_data, output)

#     # Test with optimizer
#     optimizer = Optimizer(0.01)
#     model.update(optimizer)
#     print("Model updated successfully!")
