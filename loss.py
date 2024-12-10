import numpy as np
import os

class Criterion(object):
    """
    Interface for loss functions
    """

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x,y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):
    """
    Softmax Loss
    """

    def __init__(self):
        self.softmax = None
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):
        self.logits = x
        self.labels = y
        x = self.logits - np.amax(self.logits, axis=1) # LogSumExp Trick for preventing overflow and underflow
        self.softmax = np.exp(x) / np.sum(np.exp(x), axis=1)
        self.loss = -np.sum(self.labels * np.log(self.softmax), axis=1)
        return self.loss

    def derivative(self):
        return self.softmax - self.labels



