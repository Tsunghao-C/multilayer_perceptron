from src.MLP.Dense import Dense
from src.MLP.loss_functions import BCE
from src.MLP.optimizer import Optimizer


class MLP:
    def __init__(self, input_shape, output_shape, out_activation: str = "softmax") -> None:
        """
        Initialize the Simple Neural Network with specified layers.

        Args:
            input_shape (tuple): A tuple of input examples, input features.
            output_shape (integer): Number of output classifications.
            out_activation: activation function for output layer
        """
        self.h_layer_1 = Dense(input_shape[1], 4, activation='sigmoid')
        self.h_layer_2 = Dense(4, 2, activation='sigmoid')
        self.output_layer = Dense(2, output_shape, out_activation)
        self.loss_function = BCE() # use Binary cross Entropy to calculate Loss

    def forward(self, x):
        """
        Perform the forward pass through the entire network.

        Args:
            x: Input data of shape (m, input_size), where m is the number of samples.

        Return:
            The output of the network after the forward pass.
        """
        # Store intermediate activation values
        self.a1 = self.h_layer_1.forward(x)

        self.a2 = self.h_layer_2.forward(self.a1)

        self.a3 = self.output_layer.forward(self.a2)

        return self.a3

    def backward(self, y_true, y_pred):
        """
        Perform the backward pass through the entire network.

        Args:
            y_true: Ground truth
            y_pred: Predicted values by the model (i.e. y_hat)
        """
        # Upstream gradient from loss w.r.t. network output (dL/dA3)
        dA3 = self.loss_function.loss_derivative(y_true, y_pred)
        # Upstream gradient from loss w.r.t. network input (dL/dz3 = dL/dA3 * dA3/dz3)
        dz3 = dA3 * self.output_layer.activation.backward(self.a3)  # error of outer layer

        # note: if any of the following case, the dL/dz3 is a special case
        # and can be simplified as y_pred - y_true
        # . 1. Loss function == Classical Cross Entropy, Output Activation == Softmax
        # . 2. Loss function == Binary Cross Entropy, Output Acitvation == Sigmoid

        dA2 = self.output_layer.backward(dz3)  # (dL/dA2)
        dz2 = dA2 * self.h_layer_2.activation.backward(self.a2)  # error of layer 2

        dA1 = self.h_layer_2.backward(dz2)
        dz1 = dA1 * self.h_layer_1.activation.backward(self.a1)

        _ = self.h_layer_1.backward(dz1)

    def update(self, optimizer: Optimizer):
        """
        Update model parameter

        Args:
            optimizer: An instance of Optimizer Class
        """
        optimizer.sgd(self.h_layer_1)
        optimizer.sgd(self.h_layer_2)
        optimizer.sgd(self.output_layer)


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
