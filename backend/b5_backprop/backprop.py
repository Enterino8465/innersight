import numpy as np

from innersight.backend.b3_network.network import relu_derivative, sigmoid_derivative
from innersight.backend.b4_loss.loss import compute_loss


def backward(network, output_grad):
    n_layers = len(network.weights)
    grads = [None] * n_layers

    da = output_grad

    for i in reversed(range(n_layers)):
        z = network.preacts_cache[i]
        W = network.weights[i]

        if i == n_layers - 1:
            dz = da * sigmoid_derivative(z)
        else:
            dz = da * relu_derivative(z)

        a_prev = network._last_input if i == 0 else network.activations_cache[i - 1]

        grads[i] = {
            'dW': a_prev.T @ dz,
            'db': dz.sum(axis=0, keepdims=True),
        }
        da = dz @ W.T

    return grads


def gradient_check(network, x, y, epsilon=1e-5):
    preds = network.forward(x)
    _, output_grad = compute_loss(preds, y)
    analytic_grads = backward(network, output_grad)

    rng = np.random.default_rng(0)
    for layer_idx in range(len(network.weights)):
        W = network.weights[layer_idx]
        rows, cols = W.shape
        for _ in range(3):
            r, c = rng.integers(0, rows), rng.integers(0, cols)

            W[r, c] += epsilon
            loss_plus, _ = compute_loss(network.forward(x), y)

            W[r, c] -= 2 * epsilon
            loss_minus, _ = compute_loss(network.forward(x), y)

            W[r, c] += epsilon  # restore

            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            analytic  = analytic_grads[layer_idx]['dW'][r, c]
            denom     = max(abs(analytic), abs(numerical), epsilon)
            rel_error = abs(analytic - numerical) / denom

            assert rel_error < 1e-4, (
                f'Gradient check failed at layer {layer_idx} W[{r},{c}]: '
                f'analytic={analytic:.6f}, numerical={numerical:.6f}, '
                f'rel_error={rel_error:.2e}'
            )


if __name__ == '__main__':
    from innersight.backend.b3_network.network import Network

    net = Network([3, 4, 1])
    x = np.random.randn(8, 3)
    y = np.array([0, 1, 0, 1, 0, 0, 1, 0], dtype=float)

    preds = net.forward(x)
    _, grad = compute_loss(preds, y)
    backward(net, grad)
    gradient_check(net, x, y)
    print('Gradient check passed')
