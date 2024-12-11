import numpy as np
class BatchNorm(object):
    def __init__(self, in_feature, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        self.x = x
        if eval:
            # Using running mean and var obtained from training
            self.norm = (x-self.running_mean)/np.sqrt(self.running_var + self.eps)
        else:
            self.mean = np.mean(x, axis=0)
            self.var = np.var(x, axis=0)
            self.norm = (x - self.mean)/np.sqrt(self.var + self.eps) # Normalizing the batch
            self.running_mean = self.alpha * self.running_mean + (1-self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1-self.alpha) * self.var

        self.out = self.norm * self.gamma + self.beta
        return self.out

    def backward(self, delta):
        dx = delta * self.gamma
        self.dgamma = np.sum(self.norm * delta, axis=0)
        self.dbeta = np.sum(delta, axis=0)
        raise NotImplemented