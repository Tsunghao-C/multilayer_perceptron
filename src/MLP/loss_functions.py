from abc import ABC, abstractmethod

import numpy as np


class ILossfunctions(ABC):
    """
    Template of a lossfunction class
    """
    @abstractmethod
    def loss(self, y, y_pred, eps):
        """
        Compute the binary cross entropy loss value.
        Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            eps: epsilon (default=1e-15)
        Returns:
            The binary cross entropy loss value as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
        """
        pass

    @abstractmethod
    def loss_derivative(self, y, a, eps):
        """
        Args:
            y: NumPy array of shape (m,1) or (m,), containing true binary labels (0 or 1).
            a: NumPy array of shape (m,1) or (m,), containing predicted probabilities (in [0, 1]).
            eps: epsilon (default=1e-15)
        Return:
            The derivative as a numpy array. a matrix of shape m * n
            return None if any error
        Raises:
            The function should not raise any exception
        """
        pass


class BCE(ILossfunctions):
    """
    Binary Cross Entropy caclulations
    """
    def loss(self, y, y_pred, eps=1e-15):
        # check if y, y_pred is in shape of (m,) or (m, 1)
        if y.ndim not in (1, 2) or y_pred.ndim not in (1, 2):
            return None

        # flatten y, y_pred to 1-D array and check the size
        y_flat = y.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)

        if y_flat.shape[0] != y_pred_flat.shape[0] or y_flat.shape[0] == 0:
            return None

        # check input values
        if np.any((y_flat != 0) & (y_flat != 1)):
            return None
        if np.any(y_pred_flat < 0) or np.any(y_pred_flat > 1):
            return None

        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred_flat, eps, 1 - eps)

        losses = -(y_flat * np.log(y_pred_clipped) + (1 - y_flat) * np.log(1 - y_pred_clipped))
        return float(np.mean(losses))

    def loss_derivative(self, y, a, eps):
        # The derivative of binary crossentropy with respect to predictions
        # ∂L/∂a = -(y/a - (1-y)/(1-a))
        # where y is the true label and a is the prediction (activation output)

        # check if y, a is in shape of (m,) or (m, 1)
        if y.ndim not in (1, 2) or a.ndim not in (1, 2):
            return None

        if y.shape[0] != a.shape[0] or y.shape[0] == 0:
            return None

        # check input values
        if np.any((y != 0) & (y != 1)):
            return None
        if np.any(a < 0) or np.any(a > 1):
            return None

        # Handle potential division by zero by clipping predictions
        a_clipped = np.clip(a, eps, 1 - eps)

        # Compute the derivative: ∂L/∂a = -(y/a - (1-y)/(1-a))
        derivative = -(y / a_clipped - ((1 - y) / (1 - a_clipped)))

        return derivative / y.shape[0]


class CrossEntropy(ILossfunctions):
    """
    Cross Entropy for multi-class classification
    """
    def loss(self, y, y_pred, eps=1e-15):
        """
        Compute the cross entropy loss value for multi-class classification.
        Args:
            y: has to be an numpy.ndarray, a matrix of shape (m, num_classes) with one-hot encoded labels.
            y_pred: has to be an numpy.ndarray, a matrix of shape (m, num_classes) with predicted probabilities.
            eps: epsilon (default=1e-15)
        Returns:
            The cross entropy loss value as a float.
            None on any error.
        """
        # Check if y, y_pred are 2D arrays
        if y.ndim != 2 or y_pred.ndim != 2:
            return None

        if y.shape != y_pred.shape or y.shape[0] == 0:
            return None

        # Check input values
        if np.any((y != 0) & (y != 1)):
            return None
        if np.any(y_pred < 0) or np.any(y_pred > 1):
            return None

        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

        # Compute cross entropy loss: -sum(y * log(y_pred))
        losses = -np.sum(y * np.log(y_pred_clipped), axis=1)
        return float(np.mean(losses))

    def loss_derivative(self, y, a, eps):
        """
        Compute the derivative of cross entropy loss.
        Args:
            y: NumPy array of shape (m, num_classes), containing true one-hot encoded labels.
            a: NumPy array of shape (m, num_classes), containing predicted probabilities.
            eps: epsilon (default=1e-15)
        Return:
            The derivative as a numpy array of shape (m, num_classes)
        """
        # Check if y, a are 2D arrays
        if y.ndim != 2 or a.ndim != 2:
            return None

        if y.shape != a.shape or y.shape[0] == 0:
            return None

        # Check input values
        if np.any((y != 0) & (y != 1)):
            return None
        if np.any(a < 0) or np.any(a > 1):
            return None

        # Handle potential division by zero by clipping predictions
        a_clipped = np.clip(a, eps, 1 - eps)

        # For cross entropy with softmax, the derivative is simply: a - y
        # This is because the softmax derivative cancels out with the cross entropy derivative
        derivative = a_clipped - y

        return derivative
