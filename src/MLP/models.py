from dataclasses import dataclass

import numpy as np

from src.MLP.Dense import Dense
from src.MLP.loss_functions import BCE, CrossEntropy
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
            lr: float = 0.01,
            es_threshold: float = 0.00005,
            batch_size: int = 8,
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
            weight_init = cfg.weights_init if cfg.weights_init else "xavier_uniform"
            self.layers.append(
                Dense(cfg.input_shape, cfg.output_shape, activation=cfg.activation,
                      weight_init=weight_init)
            )

        # cache of activations (post-activation outputs) per layer from last forward pass
        self._activations: list = []

        # Choose loss function based on output layer size
        output_size = network_config[-1].output_shape
        if output_size > 1:
            self.loss_function = CrossEntropy()  # Multi-class classification
        else:
            self.loss_function = BCE()  # Binary classification
        self.optimizer = Optimizer(lr)
        self.epochs = epoch
        self.batch_size = batch_size
        self.print_freq = print_freq
        self.es_threshold = es_threshold

    def forward(self, x):
        """
        Perform the forward pass through the entire network.

        Args:
            x: Input data of shape (m, input_size), where m is the number of samples.

        Return:
            The output of the network after the forward pass.
        """
        # Ensure input is a numpy array
        if hasattr(x, 'values'):  # pandas DataFrame/Series
            x = x.values
        elif not isinstance(x, np.ndarray):
            x = np.array(x)

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
        dA = self.loss_function.loss_derivative(y_true, y_pred, eps=1e-15)

        # Backpropagate through layers in reverse order
        # Convert dA to dZ for the last layer using its activation derivative
        last_idx = len(self.layers) - 1
        dz = dA * self.layers[last_idx].activation.backward(self._activations[last_idx])

        # note: if any of the following case, the dL/dz3 is a special case of y_pred - y_true
        # . 1. Loss function == Classical Cross Entropy, Output Activation == Softmax
        # . 2. Loss function == Binary Cross Entropy, Output Acitvation == Sigmoid

        # dz = y_pred - y_true
        # last_idx = len(self.layers) - 1

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

    def fit(self, x, y):
        """
        training the model
        """
        prev_loss = float("inf")
        es_buffer = 5
        es_wait = 0
        batches = int(x.shape[0] // self.batch_size + 1) if x.shape[0] % self.batch_size else int(x.shape[0] // self.batch_size)
        print(f"{batches} batches for each epoch. Batch size = {self.batch_size}")
        for epoch in range(self.epochs):
            # train data by batches
            for i in range(batches):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                x_batch = x[start:end] if i < batches - 1 else x[start:]
                y_batch = y[start:end] if i < batches - 1 else y[start:]
                # forward
                y_hat = self.forward(x_batch)
                # loss calculation
                loss = self.loss_function.loss(y_batch, y_hat, eps=1e-15)
                # print(f"Loss of batch {i} in epoch {epoch}:\t{loss}")
                # backward propogate dW, dB in each layer
                self.backward(y_batch, y_hat)
                # update weights and biases with optimizer
                self.update(self.optimizer)
            # Print epoch status
            if epoch % self.print_freq == 0:
                print(f"Epoch: {epoch}")
                print(f"loss: {loss}")
                print("=" * 30)
            # Early Stopping check
            if prev_loss - loss < self.es_threshold:
                es_wait += 1
            if es_wait >= es_buffer:
                print(f"Early stopping triggered at epoch {epoch}")
                break
            prev_loss = loss
        # save final loss in this object
        self.train_loss = loss
        return

    def predict(self, x):
        """
        predict with current network weights
        """
        return self.forward(x)

    def evaluate(self, y_pred, y_truth):
        """
        Accept the prediction, transform probability to 1 or 0 by
        choosing the biggest number as 1 else zeros.
        Then, calculate errors and accuracy.
        Args:
            y_pred: prediction generated by predict method.
            y_truth: ground truth
        """
        # Ensure inputs are numpy arrays
        if hasattr(y_pred, 'values'):
            y_pred = y_pred.values
        if hasattr(y_truth, 'values'):
            y_truth = y_truth.values

        # Convert probabilities to binary predictions
        # For each row, set the class with highest probability to 1, others to 0
        y_pred_binary = np.zeros_like(y_pred)
        y_pred_binary[np.arange(len(y_pred)), np.argmax(y_pred, axis=1)] = 1

        # Calculate accuracy
        # For one-hot encoded labels, we can compare directly
        correct_predictions = np.sum(np.all(y_pred_binary == y_truth, axis=1))
        total_predictions = len(y_truth)
        accuracy = correct_predictions / total_predictions

        # Calculate errors
        errors = total_predictions - correct_predictions
        error_rate = errors / total_predictions

        return {
            'accuracy': accuracy,
            'error_rate': error_rate,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'errors': errors
        }

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
