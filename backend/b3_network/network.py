import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(x.dtype)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1.0 - s)


class Network:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.weights.append(np.random.randn(in_size, out_size) * np.sqrt(1.0 / in_size))
            self.biases.append(np.zeros((1, out_size)))
        self.activations_cache = []
        self.preacts_cache = []

    def forward(self, x):
        self._last_input = x
        self.activations_cache = []
        self.preacts_cache = []

        current = x
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            z = current @ W + b
            a = relu(z)
            self.preacts_cache.append(z)
            self.activations_cache.append(a)
            current = a

        z_out = current @ self.weights[-1] + self.biases[-1]
        out = sigmoid(z_out)
        self.preacts_cache.append(z_out)
        self.activations_cache.append(out)

        return out
