import numpy as np


class Network:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.zeros((layer_sizes[i], layer_sizes[i + 1])))
        print(f'Network init | layers={layer_sizes} | weight_shapes={[w.shape for w in self.weights]}')

    def forward(self, x):
        output_size = self.layer_sizes[-1]
        batch_size = x.shape[0]
        out = np.random.randn(batch_size, output_size)
        print(f'forward | in={x.shape} | out={out.shape}')
        return out
