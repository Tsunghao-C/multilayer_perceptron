from abc import ABC, abstractmethod

import numpy as np

from src.MLP.Dense import Dense


class IOptimizer(ABC):
    """
    Template of an Optimizer Class
    """
    @abstractmethod
    def update(self, layer: Dense):
        """
        Update layer parameters using the optimizer's algorithm.

        Args:
            layer: Dense layer with computed gradients (dW, dB)
        """
        pass


class SGD(IOptimizer):
    """
    Stochastic Gradient Descent optimizer.
    """
    def __init__(self, lr: float = 0.01):
        """
        Initialize SGD optimizer.

        Args:
            lr: Learning rate
        """
        self.lr = lr

    def update(self, layer: Dense):
        """
        Perform Stochastic Gradient Descent optimization.

        Args:
            layer: Dense layer with computed gradients (dW, dB)
        """
        # Expect layer to have computed dW and dB in backward
        layer.weights = layer.weights - self.lr * layer.dW
        layer.biases = layer.biases - self.lr * layer.dB


class Adam(IOptimizer):
    """
    Adam optimizer with adaptive learning rates and momentum.
    """
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize Adam optimizer.

        Args:
            lr: Learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step counter

        # Initialize moment estimates (will be created per layer)
        self.m_w = {}  # First moment estimates for weights
        self.v_w = {}  # Second moment estimates for weights
        self.m_b = {}  # First moment estimates for biases
        self.v_b = {}  # Second moment estimates for biases

    def update(self, layer: Dense):
        """
        Perform Adam optimization.

        Args:
            layer: Dense layer with computed gradients (dW, dB)
        """
        # Get layer ID for tracking moment estimates
        layer_id = id(layer)

        # Initialize moment estimates for this layer if not exists
        if layer_id not in self.m_w:
            self.m_w[layer_id] = np.zeros_like(layer.weights)
            self.v_w[layer_id] = np.zeros_like(layer.weights)
            self.m_b[layer_id] = np.zeros_like(layer.biases)
            self.v_b[layer_id] = np.zeros_like(layer.biases)

        # Increment time step
        self.t += 1

        # Update biased first moment estimate for weights
        self.m_w[layer_id] = self.beta1 * self.m_w[layer_id] + (1 - self.beta1) * layer.dW

        # Update biased second moment estimate for weights
        self.v_w[layer_id] = self.beta2 * self.v_w[layer_id] + (1 - self.beta2) * (layer.dW ** 2)

        # Update biased first moment estimate for biases
        self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1 - self.beta1) * layer.dB

        # Update biased second moment estimate for biases
        self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1 - self.beta2) * (layer.dB ** 2)

        # Compute bias-corrected first moment estimate for weights
        m_w_hat = self.m_w[layer_id] / (1 - self.beta1 ** self.t)

        # Compute bias-corrected second moment estimate for weights
        v_w_hat = self.v_w[layer_id] / (1 - self.beta2 ** self.t)

        # Compute bias-corrected first moment estimate for biases
        m_b_hat = self.m_b[layer_id] / (1 - self.beta1 ** self.t)

        # Compute bias-corrected second moment estimate for biases
        v_b_hat = self.v_b[layer_id] / (1 - self.beta2 ** self.t)

        # Update weights
        layer.weights = layer.weights - self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

        # Update biases
        layer.biases = layer.biases - self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)


class RMSprop(IOptimizer):
    """
    RMSprop optimizer with exponential moving average of squared gradients.
    """
    def __init__(self, lr: float = 0.001, rho: float = 0.9, epsilon: float = 1e-8):
        """
        Initialize RMSprop optimizer.

        Args:
            lr: Learning rate
            rho: Decay rate for moving average of squared gradients
            epsilon: Small constant for numerical stability
        """
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon

        # Initialize squared gradient estimates (will be created per layer)
        self.s_w = {}  # Squared gradient estimates for weights
        self.s_b = {}  # Squared gradient estimates for biases

    def update(self, layer: Dense):
        """
        Perform RMSprop optimization.

        Args:
            layer: Dense layer with computed gradients (dW, dB)
        """
        # Get layer ID for tracking squared gradient estimates
        layer_id = id(layer)

        # Initialize squared gradient estimates for this layer if not exists
        if layer_id not in self.s_w:
            self.s_w[layer_id] = np.zeros_like(layer.weights)
            self.s_b[layer_id] = np.zeros_like(layer.biases)

        # Update squared gradient estimates for weights
        self.s_w[layer_id] = self.rho * self.s_w[layer_id] + (1 - self.rho) * (layer.dW ** 2)

        # Update squared gradient estimates for biases
        self.s_b[layer_id] = self.rho * self.s_b[layer_id] + (1 - self.rho) * (layer.dB ** 2)

        # Update weights
        layer.weights = layer.weights - self.lr * layer.dW / (np.sqrt(self.s_w[layer_id]) + self.epsilon)

        # Update biases
        layer.biases = layer.biases - self.lr * layer.dB / (np.sqrt(self.s_b[layer_id]) + self.epsilon)


def optimizer_getter(opt_type: str, lr: float = 0.01, **kwargs) -> IOptimizer:
    """
    Return an Optimizer object by type name.

    Args:
        opt_type: String name of the optimizer
        lr: Learning rate
        **kwargs: Additional optimizer-specific parameters

    Returns:
        Optimizer instance

    Raises:
        ValueError: If the opt_type is not recognized
    """
    opt_type = opt_type.lower()

    if opt_type == "sgd":
        return SGD(lr=lr)
    elif opt_type == "adam":
        return Adam(lr=lr, **kwargs)
    elif opt_type == "rmsprop":
        return RMSprop(lr=lr, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}. "
                        f"Available options: ['sgd', 'adam', 'rmsprop']")
