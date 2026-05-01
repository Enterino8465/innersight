import numpy as np


def backward(network, output_grad):
    grads = []
    for W in network.weights:
        dW = np.zeros_like(W)
        db = np.zeros(W.shape[1])
        grads.append({'dW': dW, 'db': db})
    assert len(grads) == len(network.weights)
    for i, g in enumerate(grads):
        print(f'backward | layer {i} | dW.shape={g["dW"].shape} | db.shape={g["db"].shape}')
    return grads
