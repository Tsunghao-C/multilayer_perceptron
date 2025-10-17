from abc import ABC, abstractmethod

import numpy as np


class IActivations(ABC):
    """
    Template of an Activation Class
    """
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x):
        pass


class Sigmoid(IActivations):
    def forward(self, x):
        """
        Args:
            x: a scalar or a numpy array.
        Return:
            a: The activation value as a scalar or a numpy array of any size.
        Raises:
            The function should not raise any exception
        """
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        """
        Args:
            x: the sigmoid(x), has to be a numpy array. a matrix of shape m * n
        Return:
            The derivative as a numpy array. a matrix of shape m * n
        Raises:
            The function should not raise any exception
        """
        return x * (1 - x)



class ReLU(IActivations):
    def forward(self, x: np.ndarray):
        """
        Compute the ReLU activation value.

        Args:
            x: has to be a numpy array. A matrix of shape (m, n)

        Returns:
            The ReLU activation value as a numpy array. A matrix of shape (m, n).
        """
        return np.where(x > 0, x, 0)

    def backward(self, x: np.ndarray):
        """
        Compute the derivative of the ReLU activation function.

        Args:
            x: has to be a numpy array. A matrix of shape (m, n).

        Return:
            The derivative as a numpy array. A matrix of shape (m, n).
        """
        return np.where(x > 0, 1, 0)


class SoftMax(IActivations):
    def forward(self, x):
        """
        Compute softmax values for each set of scores in x.

        Args:
            x: Input array of shape (batch_size, num_classes) or (num_classes,)

        Return:
            Softmax probablities of same shape as input
        """
        # shift first prevents overflow when calculating exp(x)
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted_x)
        sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
        prob = exp_x / sum_exp_x
        return prob

    def backward(self, s):
        """
        Args:
            s: the calculated softmax(x)
        Return:
            derivative matrix
        """
        B, C = s.shape

        # diag(s) part: each row gets its diagonal softmax values
        diag_s = np.einsum('bi,ij->bij', s, np.eye(C))  # (B, C, C)

        # outer product s s^T
        outer = np.einsum('bi,bj->bij', s, s)  # (B, C, C)

        # J = diag(s) - s s^T
        jacobian = diag_s - outer
        return jacobian



def activation_getter(act_type: str) -> IActivations:
    """
    return an Activation object by type name

    available activations:
        - Sigmoid (default)
        - ReLU
        - SoftMax
    """
    act_type = act_type.lower()
    if act_type == "relu":
        return ReLU()
    elif act_type == "softmax":
        return SoftMax()
    else:
        return Sigmoid()

# if __name__ == "__main__":
#     # Example 1: Basic sigmoid values (0.5)
#     x_1 = np.array([[0.5], [0.5], [0.5]]) # shape (3,1)
#     sigmoid = Sigmoid()
#     y_hat_1 = sigmoid.backward(x_1)
#     print("Example 1 output:\n", y_hat_1)
#     # Expected: shape (3,1)
#     # Value: [[0.25], [0.25], [0.25]] (since 0.5 * (1 - 0.5) = 0.25)
#     # Example 2: Extreme values (0 and 1)
#     x_2 = np.array([[0], [1]]) # shape (2,1)
#     sigmoid = Sigmoid()
#     y_hat_2 = sigmoid.backward(x_2)
#     print("Example 2 output:\n", y_hat_2)
#     # Expected: shape (2,1)
#     # Value: [[0], [0]] (since derivative at 0 and 1 is exactly 0)
#     # Example 3: Small and large numbers
#     x_3 = np.array([[1e-5], [0.99999]]) # shape (2,1)
#     sigmoid = Sigmoid()
#     y_hat_3 = sigmoid.backward(x_3)
#     print("Example 3 output:\n", y_hat_3)
#     # Expected: shape (2,1)
#     # [[9.9999e-06] [9.9999e-06]]
