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
        return np.ones(self.state.shape)


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
        self.state = np.where(x < 0, 0, x)
        return self.state

    def derivative(self):
       return np.where(self.state > 0, 1, 0)



def main():
    """
    ────────────────────────────────────────────────────────────────────────────────────
    ## Identity
    ────────────────────────────────────────────────────────────────────────────────────
    """

    print("──────────────────────────────────────────")
    print("IDENTITY | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    Z = np.array([
        [-4, -3],
        [-2, -1],
        [0, 1],
        [2, 3]], dtype="f")

    identity = Identity()

    A = identity.forward(Z)
    print("\nA =\n", A, sep="")

    dAdZ = identity.derivative()
    print("\ndAdZ =\n", dAdZ, sep="")

    print("\n──────────────────────────────────────────")
    print("IDENTITY | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    A_solution = np.array([
        [-4., -3.],
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")

    dAdZ_solution = np.array([
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]], dtype="f")

    print("\nA =\n", A_solution, sep="")
    print("\ndAdZ =\n", dAdZ_solution, sep="")

    print("\n──────────────────────────────────────────")
    print("IDENTITY | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("           Pass?")

    TEST_identity_A = np.allclose(A.round(4), A_solution)
    print("Test A:   ", TEST_identity_A)

    TEST_identity_dAdZ = np.allclose(dAdZ.round(4), dAdZ_solution)
    print("Test dAdZ:", TEST_identity_dAdZ)



    """
    ────────────────────────────────────────────────────────────────────────────────────
    ## Sigmoid
    ────────────────────────────────────────────────────────────────────────────────────
    """



    print("\n──────────────────────────────────────────")
    print("SIGMOID | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    Z = np.array([
        [-4, -3],
        [-2, -1],
        [0, 1],
        [2, 3]], dtype="f")

    sigmoid = Sigmoid()

    A = sigmoid.forward(Z)
    print("\nA =\n", A, sep="")

    dAdZ = sigmoid.derivative()
    print("\ndAdZ =\n", dAdZ, sep="")

    print("──────────────────────────────────────────")
    print("SIGMOID | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    A_solution = np.array([
        [0.018, 0.0474],
        [0.1192, 0.2689],
        [0.5, 0.7311],
        [0.8808, 0.9526]], dtype="f")

    dAdZ_solution = np.array([
        [0.0177, 0.0452],
        [0.105, 0.1966],
        [0.25, 0.1966],
        [0.105, 0.0452]], dtype="f")

    print("\nA =\n", A_solution, sep="")

    print("\ndAdZ =\n", dAdZ_solution, sep="")

    print("\n──────────────────────────────────────────")
    print("SIGMOID | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("\n           Pass?")

    TEST_sigmoid_A = np.allclose(A.round(4), A_solution)
    print("Test A:   ", TEST_sigmoid_A)

    TEST_sigmoid_dAdZ = np.allclose(dAdZ.round(4), dAdZ_solution)
    print("Test dAdZ:", TEST_sigmoid_dAdZ)

    """
    ────────────────────────────────────────────────────────────────────────────────────
    ## Tanh
    ────────────────────────────────────────────────────────────────────────────────────
    """


    print("\n──────────────────────────────────────────")
    print("TANH | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    Z = np.array([
        [-4, -3],
        [-2, -1],
        [0, 1],
        [2, 3]], dtype="f")

    tanh = Tanh()

    A = tanh.forward(Z)
    print("\nA =\n", A, sep="")

    dAdZ = tanh.derivative()
    print("\ndAdZ =\n", dAdZ, sep="")

    print("──────────────────────────────────────────")
    print("TANH | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    A_solution = np.array([
        [-0.9993, -0.9951],
        [-0.964, -0.7616],
        [0., 0.7616],
        [0.964, 0.9951]], dtype="f")

    dAdZ_solution = np.array([
        [0.0013, 0.0099],
        [0.0707, 0.42],
        [1., 0.42],
        [0.0707, 0.0099]], dtype="f")

    print("\nA =\n", A_solution, sep="")
    print("\ndAdZ =\n", dAdZ_solution, sep="")

    print("\n──────────────────────────────────────────")
    print("TANH | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("\n           Pass?")

    TEST_tanh_A = np.allclose(A.round(4), A_solution)
    print("Test A:   ", TEST_tanh_A)

    TEST_tanh_dAdZ = np.allclose(dAdZ.round(4), dAdZ_solution)
    print("Test dAdZ:", TEST_tanh_dAdZ)


    """
    ────────────────────────────────────────────────────────────────────────────────────
    ## ReLU
    ────────────────────────────────────────────────────────────────────────────────────
    """

    print("\n──────────────────────────────────────────")
    print("RELU | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    Z = np.array([
        [-4, -3],
        [-2, -1],
        [0, 1],
        [2, 3]], dtype="f")

    relu = ReLu()

    A = relu.forward(Z).copy()
    print("A =\n", A, sep="")

    dAdZ = relu.derivative()
    print("\ndAdZ =\n", dAdZ, sep="")

    print("\n──────────────────────────────────────────")
    print("RELU | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    A_solution = np.array([
        [0., 0.],
        [0., 0.],
        [0., 1.],
        [2., 3.]], dtype="f")

    dAdZ_solution = np.array([
        [0., 0.],
        [0., 0.],
        [0., 1.],
        [1., 1.]], dtype="f")

    print("\nA =\n", A_solution, "\n", sep="")
    print("\ndAdZ =\n", dAdZ_solution, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("RELU | CLOSENESS TEST RESULT")
    print("──────────────────────────────────────────")

    print("\n           Pass?")

    TEST_relu_A = np.allclose(A.round(4), A_solution)
    print("Test A:   ", TEST_relu_A)

    TEST_relu_dAdZ = np.allclose(dAdZ.round(4), dAdZ_solution)
    print("Test dAdZ:", TEST_relu_dAdZ)


if __name__ == "__main__":
    main()
