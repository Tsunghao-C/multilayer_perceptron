import json
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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
            es_threshold: float = 0.0002,
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

    def fit(self, x, y, save_history=True):
        """
        training the model

        Args:
            x: Training features
            y: Training labels
            save_history: Whether to save training history to CSV file
        """
        prev_loss = float("inf")
        es_buffer = 5
        es_wait = 0
        batches = int(x.shape[0] // self.batch_size + 1) if x.shape[0] % self.batch_size else int(x.shape[0] // self.batch_size)
        print(f"{batches} batches for each epoch. Batch size = {self.batch_size}")

        # Initialize training history tracking
        training_history = []

        for epoch in range(self.epochs):
            epoch_losses = []

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
                epoch_losses.append(loss)
                # print(f"Loss of batch {i} in epoch {epoch}:\t{loss}")
                # backward propogate dW, dB in each layer
                self.backward(y_batch, y_hat)
                # update weights and biases with optimizer
                self.update(self.optimizer)

            # Calculate average loss for this epoch
            avg_loss = np.mean(epoch_losses)

            # Calculate accuracy for this epoch (on full training set)
            y_pred_full = self.predict(x)
            evaluation = self.evaluate(y_pred_full, y)
            accuracy = evaluation['accuracy']

            # Record training history
            training_history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': accuracy
            })

            # Print epoch status
            if epoch % self.print_freq == 0:
                print(f"Epoch: {epoch}")
                print(f"loss: {avg_loss}")
                print(f"accuracy: {accuracy:.4f}")
                print("=" * 30)

            # Early Stopping check
            if prev_loss - avg_loss < self.es_threshold:
                es_wait += 1
            if es_wait >= es_buffer:
                print(f"Early stopping triggered at epoch {epoch}")
                break
            prev_loss = avg_loss

        # save final loss in this object
        self.train_loss = avg_loss

        # Save training history to CSV
        if save_history:
            self._save_training_history(training_history)

        return

    def _save_training_history(self, training_history):
        """
        Save training history to CSV file in trainings/ directory.

        Args:
            training_history: List of dictionaries containing epoch, loss, accuracy
        """
        # Create trainings directory if it doesn't exist
        trainings_dir = Path("trainings")
        trainings_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"history_{timestamp}.csv"
        filepath = trainings_dir / filename

        # Create DataFrame and save to CSV
        history_df = pd.DataFrame(training_history)
        history_df.to_csv(filepath, index=False)

        print(f"Training history saved to {filepath}")
        return filepath

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

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a zip file.

        Args:
            filepath: Path where to save the model (without extension, .zip will be added)
        """
        # Ensure filepath has .zip extension
        if not filepath.endswith('.zip'):
            filepath += '.zip'

        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Prepare model configuration
        config_data = {
            'epochs': self.epochs,
            'lr': self.optimizer.lr,
            'batch_size': self.batch_size,
            'print_freq': self.print_freq,
            'es_threshold': self.es_threshold,
            'loss_function': type(self.loss_function).__name__,
            'layers': []
        }

        # Prepare layer configurations
        for layer in self.layers:
            layer_config = {
                'input_size': layer.weights.shape[0],
                'output_size': layer.weights.shape[1],
                'activation': layer.activation.__class__.__name__,
                'weights_init': 'xavier_uniform'  # Default, could be enhanced to store actual init method
            }
            config_data['layers'].append(layer_config)

        # Save everything to zip file
        with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Save configuration as JSON
            config_json = json.dumps(config_data, indent=2)
            zipf.writestr('config.json', config_json)

            # Save weights and biases for each layer
            for i, layer in enumerate(self.layers):
                # Save weights
                weights_bytes = layer.weights.tobytes()
                zipf.writestr(f'layer_{i}_weights.npy', weights_bytes)

                # Save biases
                biases_bytes = layer.biases.tobytes()
                zipf.writestr(f'layer_{i}_biases.npy', biases_bytes)

        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'MLP':
        """
        Load a trained model from a zip file.

        Args:
            filepath: Path to the model zip file (with or without .zip extension)

        Returns:
            Loaded MLP model instance
        """
        # Ensure filepath has .zip extension
        if not filepath.endswith('.zip'):
            filepath += '.zip'

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load from zip file
        with zipfile.ZipFile(filepath, 'r') as zipf:
            # Load configuration
            if 'config.json' not in zipf.namelist():
                raise FileNotFoundError("config.json not found in model zip file")

            config_data = json.loads(zipf.read('config.json').decode('utf-8'))

            # Reconstruct layer configurations
            network_config = []
            for layer_config in config_data['layers']:
                dense_config = DenseConfig(
                    input_shape=layer_config['input_size'],
                    output_shape=layer_config['output_size'],
                    activation=layer_config['activation'].lower().replace('activation', ''),
                    weights_init=layer_config.get('weights_init', 'xavier_uniform')
                )
                network_config.append(dense_config)

            # Create MLP instance
            mlp = cls(
                network_config=network_config,
                epoch=config_data['epochs'],
                lr=config_data['lr'],
                batch_size=config_data['batch_size'],
                print_freq=config_data['print_freq'],
                es_threshold=config_data['es_threshold']
            )

            # Load weights and biases for each layer
            for i, layer in enumerate(mlp.layers):
                weights_filename = f'layer_{i}_weights.npy'
                biases_filename = f'layer_{i}_biases.npy'

                if weights_filename not in zipf.namelist() or biases_filename not in zipf.namelist():
                    raise FileNotFoundError(f"Weight or bias file not found for layer {i}")

                # Load weights
                weights_bytes = zipf.read(weights_filename)
                layer.weights = np.frombuffer(weights_bytes, dtype=layer.weights.dtype).reshape(layer.weights.shape)

                # Load biases
                biases_bytes = zipf.read(biases_filename)
                layer.biases = np.frombuffer(biases_bytes, dtype=layer.biases.dtype).reshape(layer.biases.shape)

        print(f"Model loaded from {filepath}")
        return mlp

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
