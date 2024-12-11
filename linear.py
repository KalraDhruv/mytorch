import numpy as np
import math

class Linear():
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):
        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)
        self.feature = None

        self.dW = np.zeros((in_feature, out_feature))
        self.db = np.zeros((1, out_feature))

        self.momentum_W = np.zeros((in_feature, out_feature))
        self.momentum_b = np.zeros((1, out_feature))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.feature = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, delta):
        self.dW = np.dot(delta, self.feature.T)
        self.db = np.mean(delta, axis=0, keepdims=True)
        dx = np.dot(delta, self.W.T)
        return dx
