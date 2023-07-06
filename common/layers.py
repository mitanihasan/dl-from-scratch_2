from common.functions import softmax, cross_entropy_error, sigmoid
import numpy as np


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        self.x = x
        return np.dot(x, W)

    def backward(self, dout):
        W, = self.params
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        dx = np.dot(dout, W.T)
        return dx


class Affine:
    def __ini__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        self.x = x
        return np.dot(x, W) + b

    def backward(self, dout):
        W, _ = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.t = None
        self.y = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(x, t)
        return loss

    def backward(self, dout=1):
        batch_size = self.y.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1  # ????
        dx *= dout
        dx = dx / batch_size

        return dx


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):

        dx = dout * self.out * (1 - self.out)
        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = sigmoid(x)
        loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)  # ????
        return loss

    def backward(self, dout):
        batch_size = self.y.shape[0]

        dx = (self.y - self.t) * dout / batch_size

        return dx
