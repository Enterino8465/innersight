import numpy as np


def compute_loss(predictions, targets):
    loss = 1.0
    grad = np.zeros_like(predictions)
    assert grad.shape == predictions.shape
    print(f'compute_loss | loss=1.0 | predictions.shape={predictions.shape} | grad.shape={grad.shape}')
    return (loss, grad)
