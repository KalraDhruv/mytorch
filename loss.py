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


class MSELoss(Criterion):
    """
    MSE Loss
    """

    def __init__(self):
        self.softmax = None
        super(MSELoss, self).__init__()

    def forward(self, x, y):
        self.logits = x
        self.labels = y
        loss_matrix = np.square(x-y)
        loss_total = np.sum(loss_matrix)
        self.mse_loss = loss_total/(2*self.logits.shape[0]*self.logits.shape[1])
        return self.mse_loss

    def backward(self):
        return (self.logits - self.labels)/(self.logits.shape[0]*self.logits.shape[1])

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
        x = self.logits - np.max(self.logits, axis=1, keepdims=True) # LogSumExp Trick for preventing overflow and underflow
        self.softmax = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        self.loss = -np.mean(np.sum(self.labels * np.log(self.softmax), axis=1))
        return self.loss

    def backward(self):
        return self.softmax - self.labels

def main():
    print("\n──────────────────────────────────────────")
    print("MSELoss | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    A = np.array([
        [-4., -3.],
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")

    Y = np.array([
        [0., 1.],
        [1., 0.],
        [1., 0.],
        [0., 1.]], dtype="f")

    mse = MSELoss()

    L = mse.forward(A, Y)
    print("\nL =\n", L.round(4), sep="")

    dLdA = mse.backward()
    print("\ndLdA =\n", dLdA, sep="")

    print("\n──────────────────────────────────────────")
    print("MSELOSS | SOLUTION OUTPUT\n")
    print("──────────────────────────────────────────")

    L_solution = np.array(3.25, dtype="f")

    dLdA_solution = np.array([
        [-0.5, -0.5],
        [-0.375, -0.125],
        [-0.125, 0.125],
        [0.25, 0.25]], dtype="f")

    print("\nL =\n", L_solution, "\n", sep="")
    print("\ndLdA =\n", dLdA_solution, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("MSELOSS | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("\n           Pass?")

    TEST_mseloss_L = np.allclose(L.round(4), L_solution)
    print("Test L:   ", TEST_mseloss_L)

    TEST_mseloss_dLdA = np.allclose(dLdA.round(4), dLdA_solution)
    print("Test dLdA:", TEST_mseloss_dLdA)


    print("\n──────────────────────────────────────────")
    print("CROSSENTROPYLOSS | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    A = np.array([
        [-4., -3.],
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")

    Y = np.array([
        [0., 1.],
        [1., 0.],
        [1., 0.],
        [0., 1.]], dtype="f")

    xent = SoftmaxCrossEntropy()

    L = xent.forward(A, Y)
    print("\nL =\n", L.round(4), sep="")

    dLdA = xent.backward()
    print("\ndLdA =\n", dLdA, sep="")

    print("──────────────────────────────────────────")
    print("CROSSENTROPYLOSS | SOLUTION OUTPUT\n")
    print("──────────────────────────────────────────")

    L_solution = np.array(0.8133, dtype="f")

    dLdA_solution = np.array([
        [0.2689, -0.2689],
        [-0.7311, 0.7311],
        [-0.7311, 0.7311],
        [0.2689, -0.2689]], dtype="f")

    print("\nL =\n", L_solution, sep="")

    print("\ndLdA =\n", dLdA_solution, sep="")

    print("\n──────────────────────────────────────────")
    print("CROSSENTROPYLOSS | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("           Pass?")

    TEST_crossentropyloss_L = np.allclose(L.round(4), L_solution)
    print("Test L:   ", TEST_crossentropyloss_L)

    TEST_crossentropyloss_dLdA = np.allclose(dLdA.round(4), dLdA_solution)
    print("Test dLdA:", TEST_crossentropyloss_dLdA)


if __name__ == "__main__":
    main()

