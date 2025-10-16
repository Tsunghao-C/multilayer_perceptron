from src.MLP.Dense import Dense


class Optimizer:
    def __init__(self, lr):
        self.lr = lr

    def sgd(self, layer: Dense):
        """
        Perform Stochastic Gradient Descent optimization.
        """
        # Expect layer to have computed dW and dB in backward
        layer.weights = layer.weights - self.lr * layer.dW
        layer.biases = layer.biases - self.lr * layer.dB


# if __name__ == "__main__":
#     import numpy as np
#     from src.MLP.activations import Sigmoid

#     # Example 1
#     x = np.array([[1, 2], [3, 4], [5, 6]]) # Shape (3,2)
#     y = np.array([[0], [1], [1]]) # Shape (3,1)
#     w = np.array([[0.1], [0.2]]) # Initial weights (2,1)
#     b = np.array([[0.5]])
#     dense = Dense(x.shape[1], y.shape[1])
#     dense.weights = w
#     dense.biases = b
#     optim = Optimizer(0.01)
#     sigmoid = Sigmoid()
#     m = len(x)
#     for i in range(m):
#         x_i = x[i, :].reshape(1, -1) # Shape (1, n)
#         y_i = y[i, :].reshape(1, 1) # Shape (1, 1)
#         # forward
#         z = dense.forward(x_i)
#         a = sigmoid.forward(z)
#         # gradient
#         loss_gradient = a - y_i
#         # backward
#         dense.backward(loss_gradient)
#         optim.sgd(dense)

#     print("Updated weight:", dense.weights.shape, dense.weights)
#     print("Updated biase: ", dense.biases.shape, dense.biases)
