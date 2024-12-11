import numpy as np
import math

class Linear():

    def __init__(self, in_feature, out_feature):
        self.W = np.zeros((in_feature, out_feature))
        self.b = np.zeros((1, out_feature))
        self.feature = None
        self.batch_size = None

        self.dW = np.zeros((in_feature, out_feature))
        self.db = np.zeros((1, out_feature))

        self.momentum_W = np.zeros((in_feature, out_feature))
        self.momentum_b = np.zeros((1, out_feature))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.batch_size = x.shape[0]
        self.feature = x
        ones = np.ones((self.batch_size, 1))
        out = np.dot(self.feature, self.W.T) + ones * self.b
        return out

    def backward(self, delta):
        self.dW = np.dot(delta.T, self.feature)
        self.db = np.mean(delta, axis=0, keepdims=True)
        dx = np.dot(delta, self.W)
        return dx
