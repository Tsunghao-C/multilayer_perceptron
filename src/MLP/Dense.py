import numpy as np

from src.MLP.activations import activation_getter


class Dense:
    def __init__(self, input_size, output_size, activation: str = "sigmoid"):
        """
        Initialize the Dense layer with weights and biases.

        Args:
            input_size: number of input features to the layer.
            output_size: number of output neurons in the layer.
            activation: activation method for this layer.
        """
        self.activation = activation_getter(activation)
        self.weights = np.random.rand(input_size, output_size)
        self.biases = np.random.rand(1, output_size)
        self.last_input = None  # cached input used for gradient computation
        self.dW = None  # gradient w.r.t. weights
        self.dB = None  # gradient w.r.t. biases
        self.z = None

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
        self.z = inputs @ self.weights + self.biases
        return self.activation.forward(self.z)

    def backward(self, gradients):
        """
        Perform the backward pass through the layer.

        Args:
            gradients: Gradient of the loss with respect to the output of this layer.

        Return:
            The gradient of the loss with respect to the input of this layer.
        """
        # gradients: dL/dA (upstream gradient w.r.t. post-activation output)
        # Convert to gradient w.r.t. pre-activation using activation derivative: dL/dZ
        dz = gradients * self.activation.backward(self.z)

        # Compute parameter gradients using cached input
        batch_size = self.last_input.shape[0]
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
