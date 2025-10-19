import numpy as np

from src.MLP.activations import activation_getter
from src.MLP.weight_init import weight_init_getter


class Dense:
    def __init__(self, input_size, output_size, activation: str = "sigmoid",
                 weight_init: str = "xavier_uniform", seed: int = 42):
        """
        Initialize the Dense layer with weights and biases.

        Args:
            input_size: number of input features to the layer.
            output_size: number of output neurons in the layer.
            activation: activation method for this layer.
            weight_init: weight initialization method ("he_uniform", "he_normal",
                        "xavier_uniform", "random_uniform").
            seed: random seed for weight initialization.
        """
        self.activation = activation_getter(activation)

        # Initialize weights using the specified weight initializer
        weight_init_class = weight_init_getter(weight_init)
        self.weight_initializer = weight_init_class(input_size, output_size, seed)
        self.weights = self.weight_initializer.gen_weights()
        self.biases = self.weight_initializer.gen_biases()

        # Cache for backward pass
        self.last_input = None  # cached input used for gradient computation
        self.dW = None  # gradient w.r.t. weights (dL/dW)
        self.dB = None  # gradient w.r.t. biases (dL/dB)

    def forward(self, inputs):
        """
        Perform the forward pass through the layer.

        Args:
            inputs: Input data of shape (m, input_size), where m is the number of samples.

        Return:
            The output of the layer after applying the weights and biases.
        """
        # cache inputs for use in backward pass
        self.last_input = inputs
        z = inputs @ self.weights + self.biases
        return self.activation.forward(z)

    def backward(self, gradients):
        """
        Perform the backward pass through the layer.

        Args:
            gradients: Gradient of the loss w.r.t. the pre-activation output of this layer. (dL/dz or Delta)

        Return:
            The gradient of the loss w.r.t. the input of this layer.
        """
        # gradients: dL/dA (upstream gradient w.r.t. post-activation output)
        # Convert to gradient w.r.t. pre-activation using activation derivative: dL/dZ = dL/dA * dA/dZ (activation_derivative)
        # dz = gradients * self.activation.backward(self.a)  # delta

        dz = gradients  # delta
        batch_size = self.last_input.shape[0]
        # Compute parameter gradients using cached input
        self.dW = self.last_input.T @ dz / batch_size
        self.dB = np.sum(dz, axis=0, keepdims=True) / batch_size

        # Gradient w.r.t. inputs: dL/dX = dL/dZ @ W^T
        return dz @ self.weights.T


# if __name__ == "__main__":
#     x = np.array([[1, 2], [3, 4], [5, 6]]) # Shape (3,2)
#     y = np.array([[0], [1], [1]]) # Shape (3, 1)
#     w = np.array([[0.1], [0.2]]) # Initial weights (2,1)
#     b = np.array([[0.5]])

#     dense = Dense(x.shape[1], y.shape[1])
#     dense.weights = w
#     dense.biases = b

#     # forward
#     z = dense.forward(x)
#     print("dense forward:", z.shape, z)

#     # gradient
#     loss_gradient = z - y
#     print("loss_gradient:", loss_gradient.shape, loss_gradient)

#     # backward
#     grads = dense.backward(loss_gradient)
#     print("dense backward:", grads.shape, grads)
#     print(dense.dW)
#     print(dense.dB)
