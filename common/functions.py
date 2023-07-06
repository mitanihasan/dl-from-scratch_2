import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    if x.dim == 1:
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))
    elif x.dim == 2:
        x = x - x.max(axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def cross_entropy_error(y, t):
    if y.dim == 1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7))
