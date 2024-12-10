import numpy as np
import os


class Activation(object):
    """
    Interface for all the activation functions
    """
    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self,x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):
    """
    Identity Function
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self,x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):
    """
    Sigmoid Function
    """
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.state = 1 / (1 + np.exp(-x))
        return self.state

    def derivative(self):
        return self.state * (1-self.state)


class Tanh(Activation):
    """
    Tanh Function
    """
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.state

    def derivative(self):
        return 1 - (self.state**2)


class ReLu(Activation):
    """
    ReLu Activation
    """
    def __init__(self):
        super(ReLu, self).__init__()

    def forward(self, x):
        if x < 0:
            self.state = 0
        else:
            self.state = x
        return self.state

    def derivative(self):
        if self.state == 0:
            return 0
        else:
            return 1