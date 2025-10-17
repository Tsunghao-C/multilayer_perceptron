from abc import ABC, abstractmethod

import numpy as np


class IWeightInitializer(ABC):
    """
    Template of a weight initializer class
    """
    def __init__(self, input_size: int, output_size: int, seed: int = 42) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.seed = seed
        np.random.seed(seed)

    @abstractmethod
    def gen_weights(self) -> np.ndarray:
        pass

    def gen_biases(self) -> np.ndarray:
        """
        Generate bias weights (typically initialized to zeros)
        """
        return np.zeros((1, self.output_size))


class HeUniform(IWeightInitializer):
    """
    He uniform weight initialization for ReLU-like activations.
    Weights are drawn from a uniform distribution: U(-sqrt(6/fan_in), sqrt(6/fan_in))
    where fan_in is the number of input units.
    """
    def __init__(self, input_size: int, output_size: int, seed: int = 42) -> None:
        super().__init__(input_size, output_size, seed)

    def gen_weights(self) -> np.ndarray:
        limit = np.sqrt(6.0 / self.input_size)
        return np.random.uniform(-limit, limit, (self.input_size, self.output_size))


class RandomUniform(IWeightInitializer):
    """
    Random uniform weight initialization.
    Weights are drawn from a uniform distribution: U(-0.1, 0.1)
    """
    def __init__(self, input_size: int, output_size: int, seed: int = 42) -> None:
        super().__init__(input_size, output_size, seed)

    def gen_weights(self) -> np.ndarray:
        return np.random.uniform(-0.1, 0.1, (self.input_size, self.output_size))


class XavierUniform(IWeightInitializer):
    """
    Xavier/Glorot uniform weight initialization.
    Weights are drawn from a uniform distribution: U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
    where fan_in is the number of input units and fan_out is the number of output units.
    Good for sigmoid and tanh activations.
    """
    def __init__(self, input_size: int, output_size: int, seed: int = 42) -> None:
        super().__init__(input_size, output_size, seed)

    def gen_weights(self) -> np.ndarray:
        limit = np.sqrt(6.0 / (self.input_size + self.output_size))
        return np.random.uniform(-limit, limit, (self.input_size, self.output_size))


class HeNormal(IWeightInitializer):
    """
    He normal weight initialization for ReLU-like activations.
    Weights are drawn from a normal distribution: N(0, sqrt(2/fan_in))
    where fan_in is the number of input units.
    """
    def __init__(self, input_size: int, output_size: int, seed: int = 42) -> None:
        super().__init__(input_size, output_size, seed)

    def gen_weights(self) -> np.ndarray:
        std = np.sqrt(2.0 / self.input_size)
        return np.random.normal(0, std, (self.input_size, self.output_size))


def weight_init_getter(init_type: str) -> type[IWeightInitializer]:
    """
    Get the appropriate weight initializer class based on the string name.

    Args:
        init_type: String name of the weight initializer

    Returns:
        The weight initializer class

    Raises:
        ValueError: If the init_type is not recognized
    """
    initializers = {
        "he_uniform": HeUniform,
        "he_normal": HeNormal,
        "xavier_uniform": XavierUniform,
        "random_uniform": RandomUniform,
    }

    if init_type not in initializers:
        raise ValueError(f"Unknown weight initializer: {init_type}. "
                        f"Available options: {list(initializers.keys())}")

    return initializers[init_type]


