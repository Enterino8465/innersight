import numpy as np


class AdamOptimizer:
    def __init__(self, network, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self.m_W = [np.zeros_like(W) for W in network.weights]
        self.v_W = [np.zeros_like(W) for W in network.weights]
        self.m_b = [np.zeros_like(b) for b in network.biases]
        self.v_b = [np.zeros_like(b) for b in network.biases]

    def step(self, network, grads):
        self.t += 1
        lr, beta1, beta2, eps, t = self.lr, self.beta1, self.beta2, self.eps, self.t

        for i in range(len(network.weights)):
            self.m_W[i] = beta1 * self.m_W[i] + (1 - beta1) * grads[i]['dW']
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * grads[i]['db']
            self.v_W[i] = beta2 * self.v_W[i] + (1 - beta2) * grads[i]['dW'] ** 2
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * grads[i]['db'] ** 2

            m_W_hat = self.m_W[i] / (1 - beta1 ** t)
            v_W_hat = self.v_W[i] / (1 - beta2 ** t)
            m_b_hat = self.m_b[i] / (1 - beta1 ** t)
            v_b_hat = self.v_b[i] / (1 - beta2 ** t)

            network.weights[i] -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            network.biases[i]  -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
