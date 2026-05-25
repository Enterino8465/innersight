import numpy as np


def compute_loss(predictions, targets, pos_weight=50.0):
    eps = 1e-7
    p = np.clip(predictions, eps, 1 - eps)
    y = targets.reshape(-1, 1)
    batch_size = len(y)

    loss_per_sample = -(pos_weight * y * np.log(p) + (1 - y) * np.log(1 - p))
    loss = float(loss_per_sample.mean())

    grad = -(pos_weight * y / p - (1 - y) / (1 - p)) / batch_size
    return loss, grad
