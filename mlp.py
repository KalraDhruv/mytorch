import numpy as np
from loss import *
from activation import *
from batchnorm import *
from linear import *

'''
Sample representation of the following functions
class Model:

    def __init__(self):
        self.layers = None
        self.activations = None

    def forward(self, A):
        l = len(self.layers)
        for i in reversed(range(1)):
            Z = None
            A = None

        return None

    def backward(self, dLdA):
        l = len(self.layers)
        for i in reversed(range(l)):
            dAdZ = None
            dLdZ = None
            dLdA = None
        return None
'''

class MLP0:

    def __init__(self):
        self.layers = [Linear(2, 3)]
        relu = ReLu()
        self.f = [relu]

    def forward(self, A0):
        self.Z0 = self.layers[0].forward(A0)
        self.A1 = self.f[0].forward(self.Z0)
        return self.A1

    def backward(self, dLdA1):
        self.dA1dZ0 = self.f[0].derivative()
        self.dLdZ0 = dLdA1 * self.dA1dZ0
        self.dLdA0 = self.layers[0].backward(self.dLdZ0)
        return self.dLdA0


class MLP1:

    def __init__(self):
        """
        Initialize 2 linear layers. Layer 1 of shape (2,3) and Layer 2 of shape (3, 2).
        Use Relu activations for both the layers.
        Implement it on the same lines(in a list) as MLP0
        """

        self.layers = [Linear(2,3), Linear(3,2)]
        self.f = [ReLu(), ReLu()]


    def forward(self, A0):

        self.Z0 = self.layers[0].forward(A0)
        self.A1 = self.f[0].forward(self.Z0)

        self.Z1 = self.layers[1].forward(self.A1)
        self.A2 = self.f[1].forward(self.Z1)

        return self.A2


    def backward(self, dLdA2):

        self.dA2dZ1 = self.f[1].derivative()
        self.dLdZ1 = dLdA2 * self.dA2dZ1
        self.dLdA1 = self.layers[1].backward(self.dLdZ1)

        self.dA1dZ0 = self.f[0].derivative()
        self.dLdZ0 =  self.dLdA1 * self.dA1dZ0
        self.dLdA0 = self.layers[0].backward(self.dLdZ0)

class MLP4:
    def __init__(self, debug=False):
        """
        Initialize 4 hidden layers and an output layer of shape below:
        Layer1 (2, 4),
        Layer2 (4, 8),
        Layer3 (8, 8),
        Layer4 (8, 4),
        Output Layer (4, 2)

        Refer the diagrmatic view in the writeup for better understanding.
        Use ReLU activation function for all the layers.)
        """
        # List of Hidden Layers
        self.layers = [Linear(2,4), Linear(4,8), Linear(8, 8),
                       Linear(8,4), Linear(4,2)]
        # List of Activations
        self.f = [ReLu(), ReLu(), ReLu(), ReLu()]

    def forward(self, A):

        self.Z = []
        self.A = [A]

        L = len(self.layers)

        for i in range(L):
            Z = self.layers[i].forward(self.A[i])
            self.Z.append(Z)
            if not i == L-1:
                A = self.f[i].forward(self.Z[i])
                self.A.append(A)

        return self.Z[L-1]

    def backward(self, dLdA):

        self.dAdZ = []
        self.dLdZ = []
        self.dLdA = [dLdA]

        L = len(self.layers)

        for i in reversed(range(L)):
            if i < L-1:
                dAdZ = self.f[i].derivative()
                dLdZ = dLdA * dAdZ
            else:
                dAdZ = np.ones_like(dLdA.shape)
                dLdZ = dLdA

            dLdA = self.layers[i].backward(dLdZ)

            self.dAdZ = [dAdZ] + self.dAdZ
            self.dLdZ = [dLdZ] + self.dLdZ
            self.dLdA = [dLdA] + self.dLdA

        return self.dLdA[0]

def main():
    """
    ────────────────────────────────────────────────────────────────────────────────────
    ## MLP0
    ────────────────────────────────────────────────────────────────────────────────────
    """
    print("\n──────────────────────────────────────────")
    print("MLP0 | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    A0 = np.array([
        [-4., -3.],
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")

    W0 = np.array([
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")

    b0 = np.array([
        [-1.],
        [0.],
        [1.]], dtype="f")

    mlp0 = MLP0()
    mlp0.layers[0].W = W0
    mlp0.layers[0].b = b0

    A1 = mlp0.forward(A0)
    Z0 = mlp0.Z0

    print("Z0 =\n", Z0.round(4), sep="")

    print("\nA1 =\n", A1.round(4), sep="")

    dLdA1 = np.array([
        [-4., -3., -2.],
        [-1., -0., 1.],
        [2., 3., 4.],
        [5., 6., 7.]], dtype="f")

    mlp0.backward(dLdA1)

    dA1dZ0 = mlp0.dA1dZ0
    print("\ndA1dZ0 =\n", dA1dZ0, sep="")

    dLdZ0 = mlp0.dLdZ0
    print("\ndLdZ0 =\n", dLdZ0, sep="")

    dLdA0 = mlp0.dLdA0
    print("\ndLdA0 =\n", dLdA0, sep="")

    dLdW0 = mlp0.layers[0].dLdW
    print("\ndLdW0 =\n", dLdW0, sep="")

    dLdb0 = mlp0.layers[0].dLdb
    print("\ndLdb0 =\n", dLdb0, sep="")

    print("──────────────────────────────────────────")
    print("MLP0 | SOLUTION OUTPUT\n")
    print("──────────────────────────────────────────")

    Z0_solution = np.array([
        [10., -3., -16.],
        [4., -1., -6.],
        [-2., 1., 4.],
        [-8., 3., 14.]], dtype="f")

    A1_solution = np.array([
        [10., 0., 0.],
        [4., 0., 0.],
        [0., 1., 4.],
        [0., 3., 14.]], dtype="f")

    dA1dZ0_solution = np.array([
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 1.],
        [0., 1., 1.]], dtype="f")

    dLdZ0_solution = np.array([
        [-4., -0., -0.],
        [-1., -0., 0.],
        [0., 3., 4.],
        [0., 6., 7.]], dtype="f")

    dLdA0_solution = np.array([
        [8., 4.],
        [2., 1.],
        [8., 15.],
        [14., 27.]], dtype="f")

    dLdW0_solution = np.array([
        [4.5, 3.25],
        [3., 5.25],
        [3.5, 6.25]], dtype="f")

    dLdb0_solution = np.array([
        [-1.25],
        [2.25],
        [2.75]], dtype="f")


    print("\n──────────────────────────────────────────")
    print("MLP0 | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("             Pass?")

    TEST_mlp0_Z0 = np.allclose(Z0.round(4), Z0_solution)
    print("Test Z0:    ", TEST_mlp0_Z0)

    TEST_mlp0_A1 = np.allclose(A1.round(4), A1_solution)
    print("Test A1:    ", TEST_mlp0_A1)

    TEST_mlp0_dA1dZ0 = np.allclose(dA1dZ0.round(4), dA1dZ0_solution)
    print("Test dA1dZ0:", TEST_mlp0_dA1dZ0)

    TEST_mlp0_dLdZ0 = np.allclose(dLdZ0.round(4), dLdZ0_solution)
    print("Test dLdZ0: ", TEST_mlp0_dLdZ0)

    TEST_mlp0_dLdA0 = np.allclose(dLdA0.round(4), dLdA0_solution)
    print("Test dLdA0: ", TEST_mlp0_dLdA0)

    TEST_mlp0_dLdW0 = np.allclose(dLdW0.round(4), dLdW0_solution)
    print("Test dLdW0: ", TEST_mlp0_dLdW0)

    TEST_mlp0_dLdb0 = np.allclose(dLdb0.round(4), dLdb0_solution)
    print("Test dLdb0: ", TEST_mlp0_dLdb0)

    """## MPL1"""

    print("──────────────────────────────────────────")
    print("MLP0 | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    A0 = np.array([
        [-4., -3.],
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")

    W0 = np.array([
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")

    b0 = np.array([
        [-1.],
        [0.],
        [1.]], dtype="f")

    W1 = np.array([
        [-2., -1., 0],
        [1., 2., 3]], dtype="f")

    b1 = np.array([
        [-1.],
        [1.]], dtype="f")

    mlp1 = MLP1()
    mlp1.layers[0].W = W0
    mlp1.layers[0].b = b0
    mlp1.layers[1].W = W1
    mlp1.layers[1].b = b1

    A2 = mlp1.forward(A0)

    Z0 = mlp1.Z0
    A1 = mlp1.A1
    Z1 = mlp1.Z1

    print("Z0 =\n", Z0.round(4), sep="")

    print("\nA1 =\n", A1.round(4), sep="")

    print("\nZ1 =\n", Z1.round(4), sep="")

    print("\nA2 =\n", A2.round(4), sep="")

    dLdA2 = np.array([
        [-4., -3.],
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")

    mlp1.backward(dLdA2)

    dA2dZ1 = mlp1.dA2dZ1
    print("\ndA2dZ1 =\n", dA2dZ1, sep="")

    dLdZ1 = mlp1.dLdZ1
    print("\ndLdZ1 =\n", dLdZ1, sep="")

    dLdA1 = mlp1.dLdA1
    print("\ndLdA1 =\n", dLdA1, sep="")

    dA1dZ0 = mlp1.dA1dZ0
    print("\ndA1dZ0 =\n", dA1dZ0, sep="")

    dLdZ0 = mlp1.dLdZ0
    print("\ndLdZ0 =\n", dLdZ0, sep="")

    dLdA0 = mlp1.dLdA0
    print("\ndLdA0 =\n", dLdA0, sep="")

    dLdW0 = mlp1.layers[0].dLdW
    print("\ndLdW0 =\n", dLdW0, sep="")

    dLdb0 = mlp1.layers[0].dLdb
    print("\ndLdb0 =\n", dLdb0, sep="")

    print("──────────────────────────────────────────")
    print("MLP1 | SOLUTION OUTPUT\n")
    print("──────────────────────────────────────────")

    Z0_solution = np.array([
        [10., -3., -16.],
        [4., -1., -6.],
        [-2., 1., 4.],
        [-8., 3., 14.]], dtype="f")

    A1_solution = np.array([
        [10., 0., 0.],
        [4., 0., 0.],
        [0., 1., 4.],
        [0., 3., 14.]], dtype="f")

    Z1_solution = np.array([
        [-21., 11.],
        [-9., 5.],
        [-2., 15.],
        [-4., 49.]], dtype="f")

    A2_solution = np.array([
        [0., 11.],
        [0., 5.],
        [0., 15.],
        [0., 49.]], dtype="f")

    dA2dZ1_solution = np.array([
        [0., 1.],
        [0., 1.],
        [0., 1.],
        [0., 1.]], dtype="f")

    dLdZ1_solution = np.array([
        [-0., -3.],
        [-0., -1.],
        [0., 1.],
        [0., 3.]], dtype="f")

    dLdA1_solution = np.array([
        [-3., -6., -9.],
        [-1., -2., -3.],
        [1., 2., 3.],
        [3., 6., 9.]], dtype="f")

    dA1dZ0_solution = np.array([
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 1.],
        [0., 1., 1.]], dtype="f")

    dLdZ0_solution = np.array([
        [-3., -0., -0.],
        [-1., -0., -0.],
        [0., 2., 3.],
        [0., 6., 9.]], dtype="f")

    dLdA0_solution = np.array([
        [6., 3.],
        [2., 1.],
        [6., 11.],
        [18., 33.]], dtype="f")

    dLdW0_solution = np.array([
        [3.5, 2.5],
        [3., 5.],
        [4.5, 7.5]], dtype="f")

    dLdb0_solution = np.array([
        [-1.],
        [2.],
        [3.]], dtype="f")

    print("\ndA2dZ1 =\n", dA2dZ1, sep="")
    print("\ndLdZ1 =\n", dLdZ1, sep="")
    print("\ndLdA1 =\n", dLdA1, sep="")
    print("\ndA1dZ0 =\n", dA1dZ0, sep="")
    print("\ndLdZ0 =\n", dLdZ0, sep="")
    print("\ndLdA0 =\n", dLdA0, sep="")
    print("\ndLdW0 =\n", dLdW0, sep="")
    print("\ndLdb0 =\n", dLdb0, sep="")

    print("\n──────────────────────────────────────────")
    print("MLP1 | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("             Pass?")

    TEST_mlp1_Z0 = np.allclose(Z0.round(4), Z0_solution)
    print("Test Z0:    ", TEST_mlp1_Z0)

    TEST_mlp1_A1 = np.allclose(A1.round(4), A1_solution)
    print("Test A1:    ", TEST_mlp1_A1)

    TEST_mlp1_Z1 = np.allclose(Z1.round(4), Z1_solution)
    print("Test Z1:    ", TEST_mlp1_Z1)

    TEST_mlp1_A2 = np.allclose(A2.round(4), A2_solution)
    print("Test A2:    ", TEST_mlp1_A2)

    TEST_mlp1_dA2dZ1 = np.allclose(dA2dZ1.round(4), dA2dZ1_solution)
    print("Test dA2dZ1:", TEST_mlp1_dA2dZ1)

    TEST_mlp1_dLdZ1 = np.allclose(dLdZ1.round(4), dLdZ1_solution)
    print("Test dLdZ1: ", TEST_mlp1_dLdZ1)

    TEST_mlp1_dLdA1 = np.allclose(dLdA1.round(4), dLdA1_solution)
    print("Test dLdA1: ", TEST_mlp1_dLdA1)

    TEST_mlp1_dA1dZ0 = np.allclose(dA1dZ0.round(4), dA1dZ0_solution)
    print("Test dA1dZ0:", TEST_mlp1_dA1dZ0)

    TEST_mlp1_dLdZ0 = np.allclose(dLdZ0.round(4), dLdZ0_solution)
    print("Test dLdZ0: ", TEST_mlp1_dLdZ0)

    TEST_mlp1_dLdA0 = np.allclose(dLdA0.round(4), dLdA0_solution)
    print("Test dLdA0: ", TEST_mlp1_dLdA0)

    TEST_mlp1_dLdW0 = np.allclose(dLdW0.round(4), dLdW0_solution)
    print("Test dLdW0: ", TEST_mlp1_dLdW0)

    TEST_mlp1_dLdb0 = np.allclose(dLdb0.round(4), dLdb0_solution)
    print("Test dLdb0: ", TEST_mlp1_dLdb0)


    """
    ────────────────────────────────────────────────────────────────────────────────────
    ## MLP4
    ────────────────────────────────────────────────────────────────────────────────────
    """

    print("\n──────────────────────────────────────────")
    print("MLP4 FORWARD | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    A0 = np.array([
        [-4., -3.],
        [-2., -1.],
        [0., 1.],
        [2., 3.],
        [4., 5.]], dtype="f")

    W0 = np.array([
        [0., 1.],
        [1., 2.],
        [2., 0.],
        [0., 1.]], dtype="f")

    b0 = np.array([
        [1.],
        [1.],
        [1.],
        [1.]], dtype="f")

    W1 = np.array([
        [0., 2., 1., 0.],
        [1., 0., 2., 1.],
        [2., 1., 0., 2.],
        [0., 2., 1., 0.],
        [1., 0., 2., 1.],
        [2., 1., 0., 2.],
        [0., 2., 1., 0.],
        [1., 0., 2., 1.]], dtype="f")

    b1 = np.array([
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.]], dtype="f")

    W2 = np.array([
        [0., 2., 1., 0., 2., 1., 0., 2.],
        [1., 0., 2., 1., 0., 2., 1., 0.],
        [2., 1., 0., 2., 1., 0., 2., 1.],
        [0., 2., 1., 0., 2., 1., 0., 2.],
        [1., 0., 2., 1., 0., 2., 1., 0.],
        [2., 1., 0., 2., 1., 0., 2., 1.],
        [0., 2., 1., 0., 2., 1., 0., 2.],
        [1., 0., 2., 1., 0., 2., 1., 0.]], dtype="f")

    b2 = np.array([
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.]], dtype="f")

    W3 = np.array([
        [0., 1., 2., 0., 1., 2., 0., 1.],
        [1., 2., 0., 1., 2., 0., 1., 2.],
        [2., 0., 1., 2., 0., 1., 2., 0.],
        [0., 1., 2., 0., 1., 2., 0., 1.]], dtype="f")

    b3 = np.array([
        [1.],
        [1.],
        [1.],
        [1.]], dtype="f")

    W4 = np.array([
        [0., 2., 1., 0.],
        [1., 0., 2., 1.]], dtype="f")

    b4 = np.array([
        [1.],
        [1.]], dtype="f")

    mlp4 = MLP4()

    mlp4.layers[0].W = W0
    mlp4.layers[0].b = b0
    mlp4.layers[1].W = W1
    mlp4.layers[1].b = b1
    mlp4.layers[2].W = W2
    mlp4.layers[2].b = b2
    mlp4.layers[3].W = W3
    mlp4.layers[3].b = b3
    mlp4.layers[4].W = W4
    mlp4.layers[4].b = b4

    A5 = mlp4.forward(A0)

    Z0 = mlp4.Z[0]
    print("\nZ0 =\n", Z0, sep="")
    A1 = mlp4.A[1]
    print("\nA1 =\n", A1, sep="")
    Z1 = mlp4.Z[1]
    print("\nZ1 =\n", Z1, sep="")
    A2 = mlp4.A[2]
    print("\nA2 =\n", A2, sep="")
    Z2 = mlp4.Z[2]
    print("\nZ2 =\n", Z2, sep="")
    A3 = mlp4.A[3]
    print("\nA3 =\n", A3, sep="")
    Z3 = mlp4.Z[3]
    print("\nZ3 =\n", Z3, sep="")
    A4 = mlp4.A[4]
    print("\nA4 =\n", A4, sep="")
    Z4 = mlp4.Z[4]
    print("\nZ4 =\n", Z4, sep="")

    print("\nA5 =\n", A5, sep="")

    print("\n──────────────────────────────────────────")
    print("MLP4 FORWARD | TEST RESULTS")
    print("──────────────────────────────────────────")

    Z0_solution = np.array([
        [-2., -9., -7., -2.],
        [0., -3., -3., 0.],
        [2., 3., 1., 2.],
        [4., 9., 5., 4.],
        [6., 15., 9., 6.]], dtype="f")

    A1_solution = np.array([
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [2., 3., 1., 2.],
        [4., 9., 5., 4.],
        [6., 15., 9., 6.]], dtype="f")

    Z1_solution = np.array([
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [8., 7., 12., 8., 7., 12., 8., 7.],
        [24., 19., 26., 24., 19., 26., 24., 19.],
        [40., 31., 40., 40., 31., 40., 40., 31.]], dtype="f")

    A2_solution = np.array([
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [8., 7., 12., 8., 7., 12., 8., 7.],
        [24., 19., 26., 24., 19., 26., 24., 19.],
        [40., 31., 40., 40., 31., 40., 40., 31.]], dtype="f")

    Z2_solution = np.array([
        [9., 8., 10., 9., 8., 10., 9., 8.],
        [9., 8., 10., 9., 8., 10., 9., 8.],
        [67., 73., 70., 67., 73., 70., 67., 73.],
        [167., 177., 202., 167., 177., 202., 167., 177.],
        [267., 281., 334., 267., 281., 334., 267., 281.]], dtype="f")

    A3_solution = np.array([
        [9., 8., 10., 9., 8., 10., 9., 8.],
        [9., 8., 10., 9., 8., 10., 9., 8.],
        [67., 73., 70., 67., 73., 70., 67., 73.],
        [167., 177., 202., 167., 177., 202., 167., 177.],
        [267., 281., 334., 267., 281., 334., 267., 281.]], dtype="f")

    Z3_solution = np.array([
        [65., 76., 75., 65.],
        [65., 76., 75., 65.],
        [500., 640., 543., 500.],
        [1340., 1564., 1407., 1340.],
        [2180., 2488., 2271., 2180.]], dtype="f")

    A4_solution = np.array([
        [65., 76., 75., 65.],
        [65., 76., 75., 65.],
        [500., 640., 543., 500.],
        [1340., 1564., 1407., 1340.],
        [2180., 2488., 2271., 2180.]], dtype="f")

    Z4_solution = np.array([
        [228., 281.],
        [228., 281.],
        [1824., 2087.],
        [4536., 5495.],
        [7248., 8903.]], dtype="f")

    A5_solution = np.array([
        [228., 281.],
        [228., 281.],
        [1824., 2087.],
        [4536., 5495.],
        [7248., 8903.]], dtype="f")

    print("\nZ0 =\n", Z0_solution, sep="")
    print("\nA1 =\n", A1_solution, sep="")
    print("\nZ1 =\n", Z1_solution, sep="")
    print("\nA2 =\n", A2_solution, sep="")
    print("\nZ2 =\n", Z2_solution, sep="")
    print("\nA3 =\n", A3_solution, sep="")
    print("\nZ3 =\n", Z3_solution, sep="")
    print("\nA4 =\n", A4_solution, sep="")
    print("\nZ4 =\n", Z4_solution, sep="")

    print("\nA5 =\n", A5_solution, sep="")

    print("\n──────────────────────────────────────────")
    print("MLP4 FORWARD | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("             Pass?")

    TEST_mlp4_Z0 = np.allclose(Z0.round(4), Z0_solution)
    print("Test Z0:    ", TEST_mlp4_Z0)

    TEST_mlp4_A1 = np.allclose(A1.round(4), A1_solution)
    print("Test A1:    ", TEST_mlp4_A1)

    TEST_mlp4_Z1 = np.allclose(Z1.round(4), Z1_solution)
    print("Test Z1:    ", TEST_mlp4_Z1)

    TEST_mlp4_A2 = np.allclose(A2.round(4), A2_solution)
    print("Test A2:    ", TEST_mlp4_A2)

    TEST_mlp4_Z2 = np.allclose(Z2.round(4), Z2_solution)
    print("Test Z2:    ", TEST_mlp4_Z2)

    TEST_mlp4_A3 = np.allclose(A3.round(4), A3_solution)
    print("Test A3:    ", TEST_mlp4_A3)

    TEST_mlp4_Z3 = np.allclose(Z3.round(4), Z3_solution)
    print("Test Z3:    ", TEST_mlp4_Z3)

    TEST_mlp4_A4 = np.allclose(A4.round(4), A4_solution)
    print("Test A4:    ", TEST_mlp4_A4)

    TEST_mlp4_Z4 = np.allclose(Z4.round(4), Z4_solution)
    print("Test Z4:    ", TEST_mlp4_Z4)

    TEST_mlp4_A5 = np.allclose(A5.round(4), A5_solution)
    print("Test A5:    ", TEST_mlp4_A5)

    print("\n──────────────────────────────────────────")
    print("MLP4 BACKWARD | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    dLdA5 = np.array([
        [-4., -3.],
        [-2., -1.],
        [0., 1.],
        [2., 3.],
        [4., 5.]], dtype="f")

    mlp4.backward(dLdA5)

    dA1dZ0 = mlp4.dAdZ[0]
    print("\ndA1dZ0 =\n", dA1dZ0, sep="")
    dLdZ0 = mlp4.dLdZ[0]
    print("\ndLdZ0 =\n", dLdZ0, sep="")
    dLdA0 = mlp4.dLdA[0]
    print("\ndLdA0 =\n", dLdA0, sep="")
    dLdW0 = mlp4.layers[0].dLdW
    print("\ndLdW0 =\n", dLdW0, sep="")
    dLdb0 = mlp4.layers[0].dLdb
    print("\ndLdb0 =\n", dLdb0, sep="")

    dA2dZ1 = mlp4.dAdZ[1]
    print("\ndA2dZ1 =\n", dA2dZ1, sep="")
    dLdZ1 = mlp4.dLdZ[1]
    print("\ndLdZ1 =\n", dLdZ1, sep="")
    dLdA1 = mlp4.dLdA[1]
    print("\ndLdA1 =\n", dLdA1, sep="")
    dLdW1 = mlp4.layers[1].dLdW
    print("\ndLdW1 =\n", dLdW1, sep="")
    dLdb1 = mlp4.layers[1].dLdb
    print("\ndLdb1 =\n", dLdb1, sep="")

    dA3dZ2 = mlp4.dAdZ[2]
    print("\ndA3dZ2 =\n", dA3dZ2, sep="")
    dLdZ2 = mlp4.dLdZ[2]
    print("\ndLdZ2 =\n", dLdZ2, sep="")
    dLdA2 = mlp4.dLdA[2]
    print("\ndLdA2 =\n", dLdA2, sep="")
    dLdW2 = mlp4.layers[2].dLdW
    print("\ndLdW2 =\n", dLdW2, sep="")
    dLdb2 = mlp4.layers[2].dLdb
    print("\ndLdb2 =\n", dLdb2, sep="")

    dA4dZ3 = mlp4.dAdZ[3]
    print("\ndA4dZ3 =\n", dA4dZ3, sep="")
    dLdZ3 = mlp4.dLdZ[3]
    print("\ndLdZ3 =\n", dLdZ3, sep="")
    dLdA3 = mlp4.dLdA[3]
    print("\ndLdA3 =\n", dLdA3, sep="")
    dLdW3 = mlp4.layers[3].dLdW
    print("\ndLdW3 =\n", dLdW3, sep="")
    dLdb3 = mlp4.layers[3].dLdb
    print("\ndLdb3 =\n", dLdb3, sep="")

    dA5dZ4 = mlp4.dAdZ[4]
    print("\ndA5dZ4 =\n", dA5dZ4, sep="")
    dLdZ4 = mlp4.dLdZ[4]
    print("\ndLdZ4 =\n", dLdZ4, sep="")
    dLdA4 = mlp4.dLdA[4]
    print("\ndLdA4 =\n", dLdA4, sep="")
    dLdW4 = mlp4.layers[4].dLdW
    print("\ndLdW4 =\n", dLdW4, sep="")
    dLdb4 = mlp4.layers[4].dLdb
    print("\ndLdb4 =\n", dLdb4, sep="")

    print("\n──────────────────────────────────────────")
    print("MLP4 BACKWARD | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    dA1dZ0_solution = np.array([
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]], dtype="f")

    dLdZ0_solution = np.array([
        [-0., -0., -0., -0.],
        [-0., -0., -0., -0.],
        [204., 228., 306., 204.],
        [1056., 1020., 1326., 1056.],
        [1908., 1812., 2346., 1908.]], dtype="f")

    dLdA0_solution = np.array([
        [0., 0.],
        [0., 0.],
        [840., 864.],
        [3672., 4152.],
        [6504., 7440.]], dtype="f")

    dLdW0_solution = np.array([
        [1948.8, 2582.4],
        [1857.6, 2469.6],
        [2407.2, 3202.8],
        [1948.8, 2582.4]], dtype="f")

    dLdb0_solution = np.array([
        [633.6],
        [612.],
        [795.6],
        [633.6]], dtype="f")

    dA2dZ1_solution = np.array([
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.]], dtype="f")

    dLdZ1_solution = np.array([
        [-154., -212., -216., -154., -212., -216., -154., -212.],
        [-62., -88., -96., -62., -88., -96., -62., -88.],
        [30., 36., 24., 30., 36., 24., 30., 36.],
        [122., 160., 144., 122., 160., 144., 122., 160.],
        [214., 284., 264., 214., 284., 264., 214., 284.]], dtype="f")

    dLdA1_solution = np.array([
        [-1500., -1356., -1734., -1500.],
        [-648., -564., -714., -648.],
        [204., 228., 306., 204.],
        [1056., 1020., 1326., 1056.],
        [1908., 1812., 2346., 1908.]], dtype="f")

    dLdW1_solution = np.array([
        [366.4, 879.6, 513.2, 366.4],
        [483.2, 1161.6, 678.4, 483.2],
        [441.6, 1065.6, 624., 441.6],
        [366.4, 879.6, 513.2, 366.4],
        [483.2, 1161.6, 678.4, 483.2],
        [441.6, 1065.6, 624., 441.6],
        [366.4, 879.6, 513.2, 366.4],
        [483.2, 1161.6, 678.4, 483.2]], dtype="f")

    dLdb1_solution = np.array([
        [30.],
        [36.],
        [24.],
        [30.],
        [36.],
        [24.],
        [30.],
        [36.]], dtype="f")

    dA3dZ2_solution = np.array([
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.]], dtype="f")

    dLdZ2_solution = np.array([
        [-28., -22., -22., -28., -22., -22., -28., -22.],
        [-12., -10., -8., -12., -10., -8., -12., -10.],
        [4., 2., 6., 4., 2., 6., 4., 2.],
        [20., 14., 20., 20., 14., 20., 20., 14.],
        [36., 26., 34., 36., 26., 34., 36., 26.]], dtype="f")

    dLdA2_solution = np.array([
        [-154., -212., -216., -154., -212., -216., -154., -212.],
        [-62., -88., -96., -62., -88., -96., -62., -88.],
        [30., 36., 24., 30., 36., 24., 30., 36.],
        [122., 160., 144., 122., 160., 144., 122., 160.],
        [214., 284., 264., 214., 284., 264., 214., 284.]], dtype="f")

    dLdW2_solution = np.array([
        [382.4, 296.8, 393.6, 382.4, 296.8, 393.6, 382.4, 296.8],
        [272., 210.8, 279.2, 272., 210.8, 279.2, 272., 210.8],
        [371.6, 289.2, 384.4, 371.6, 289.2, 384.4, 371.6, 289.2],
        [382.4, 296.8, 393.6, 382.4, 296.8, 393.6, 382.4, 296.8],
        [272., 210.8, 279.2, 272., 210.8, 279.2, 272., 210.8],
        [371.6, 289.2, 384.4, 371.6, 289.2, 384.4, 371.6, 289.2],
        [382.4, 296.8, 393.6, 382.4, 296.8, 393.6, 382.4, 296.8],
        [272., 210.8, 279.2, 272., 210.8, 279.2, 272., 210.8]], dtype="f")

    dLdb2_solution = np.array([
        [4.],
        [2.],
        [6.],
        [4.],
        [2.],
        [6.],
        [4.],
        [2.]], dtype="f")

    dA4dZ3_solution = np.array([
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]], dtype="f")

    dLdZ3_solution = np.array([
        [-3., -8., -10., -3.],
        [-1., -4., -4., -1.],
        [1., 0., 2., 1.],
        [3., 4., 8., 3.],
        [5., 8., 14., 5.]], dtype="f")

    dLdA3_solution = np.array([
        [-28., -22., -22., -28., -22., -22., -28., -22.],
        [-12., -10., -8., -12., -10., -8., -12., -10.],
        [4., 2., 6., 4., 2., 6., 4., 2.],
        [20., 14., 20., 20., 14., 20., 20., 14.],
        [36., 26., 34., 36., 26., 34., 36., 26.]], dtype="f")

    dLdW3_solution = np.array([
        [373.4, 395.4, 461.2, 373.4, 395.4, 461.2, 373.4, 395.4],
        [539.2, 572., 672., 539.2, 572., 672., 539.2, 572.],
        [1016.4, 1076.8, 1258.4, 1016.4, 1076.8, 1258.4, 1016.4, 1076.8],
        [373.4, 395.4, 461.2, 373.4, 395.4, 461.2, 373.4, 395.4]], dtype="f")

    dLdb3_solution = np.array([
        [1.],
        [0.],
        [2.],
        [1.]], dtype="f")

    dA5dZ4_solution = np.array([
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]], dtype="f")

    dLdZ4_solution = np.array([
        [-4., -3.],
        [-2., -1.],
        [0., 1.],
        [2., 3.],
        [4., 5.]], dtype="f")

    dLdA4_solution = np.array([
        [-3., -8., -10., -3.],
        [-1., -4., -4., -1.],
        [1., 0., 2., 1.],
        [3., 4., 8., 3.],
        [5., 8., 14., 5.]], dtype="f")

    dLdW4_solution = np.array([
        [2202., 2524.8, 2289.6, 2202.],
        [3032., 3493.6, 3163.8, 3032.]], dtype="f")

    dLdb4_solution = np.array([
        [0.],
        [1.]], dtype="f")

    print("\ndA1dZ0 =\n", dA1dZ0_solution, sep="")
    print("\ndLdZ0 =\n", dLdZ0_solution, sep="")
    print("\ndLdA0 =\n", dLdA0_solution, sep="")
    print("\ndLdW0 =\n", dLdW0_solution, sep="")
    print("\ndLdb0 =\n", dLdb0_solution, sep="")

    print("\ndA2dZ1 =\n", dA2dZ1_solution, sep="")
    print("\ndLdZ1 =\n", dLdZ1_solution, sep="")
    print("\ndLdA1 =\n", dLdA1_solution, sep="")
    print("\ndLdW1 =\n", dLdW1_solution, sep="")
    print("\ndLdb1 =\n", dLdb1_solution, sep="")

    print("\ndA3dZ2 =\n", dA3dZ2_solution, sep="")
    print("\ndLdZ2 =\n", dLdZ2_solution, sep="")
    print("\ndLdA2 =\n", dLdA2_solution, sep="")
    print("\ndLdW2 =\n", dLdW2_solution, sep="")
    print("\ndLdb2 =\n", dLdb2_solution, sep="")

    print("\ndA4dZ3 =\n", dA4dZ3_solution, sep="")
    print("\ndLdZ3 =\n", dLdZ3_solution, sep="")
    print("\ndLdA3 =\n", dLdA3_solution, sep="")
    print("\ndLdW3 =\n", dLdW3_solution, sep="")
    print("\ndLdb3 =\n", dLdb3_solution, sep="")

    print("\ndA5dZ4 =\n", dA5dZ4_solution, sep="")
    print("\ndLdZ4 =\n", dLdZ4_solution, sep="")
    print("\ndLdA4 =\n", dLdA4_solution, sep="")
    print("\ndLdW4 =\n", dLdW4_solution, sep="")
    print("\ndLdb4 =\n", dLdb4_solution, sep="")

    print("\n──────────────────────────────────────────")
    print("MLP4 BACKWARD | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("\n             Pass?")

    TEST_mlp4_dA1dZ0 = np.allclose(dA1dZ0.round(4), dA1dZ0_solution)
    print("Test dA1dZ0:", TEST_mlp4_dA1dZ0)

    TEST_mlp4_dLdZ0 = np.allclose(dLdZ0.round(4), dLdZ0_solution)
    print("Test dLdZ0: ", TEST_mlp4_dLdZ0)

    TEST_mlp4_dLdA0 = np.allclose(dLdA0.round(4), dLdA0_solution)
    print("Test dLdA0: ", TEST_mlp4_dLdA0)

    TEST_mlp4_dLdW0 = np.allclose(dLdW0.round(4), dLdW0_solution)
    print("Test dLdW0: ", TEST_mlp4_dLdW0)

    TEST_mlp4_dLdb0 = np.allclose(dLdb0.round(4), dLdb0_solution)
    print("Test dLdb0: ", TEST_mlp4_dLdb0)

    TEST_mlp4_dA2dZ1 = np.allclose(dA2dZ1.round(4), dA2dZ1_solution)
    print("\nTest dA2dZ1:", TEST_mlp4_dA2dZ1)

    TEST_mlp4_dLdZ1 = np.allclose(dLdZ1.round(4), dLdZ1_solution)
    print("Test dLdZ1: ", TEST_mlp4_dLdZ1)

    TEST_mlp4_dLdA1 = np.allclose(dLdA1.round(4), dLdA1_solution)
    print("Test dLdA1: ", TEST_mlp4_dLdA1)

    TEST_mlp4_dLdW1 = np.allclose(dLdW1.round(4), dLdW1_solution)
    print("Test dLdW1: ", TEST_mlp4_dLdW1)

    TEST_mlp4_dLdb1 = np.allclose(dLdb1.round(4), dLdb1_solution)
    print("Test dLdb1: ", TEST_mlp4_dLdb1)

    TEST_mlp4_dA3dZ2 = np.allclose(dA3dZ2.round(4), dA3dZ2_solution)
    print("\nTest dA3dZ2:", TEST_mlp4_dA3dZ2)

    TEST_mlp4_dLdZ2 = np.allclose(dLdZ2.round(4), dLdZ2_solution)
    print("Test dLdZ2: ", TEST_mlp4_dLdZ2)

    TEST_mlp4_dLdA2 = np.allclose(dLdA2.round(4), dLdA2_solution)
    print("Test dLdA2: ", TEST_mlp4_dLdA2)

    TEST_mlp4_dLdW2 = np.allclose(dLdW2.round(4), dLdW2_solution)
    print("Test dLdW2: ", TEST_mlp4_dLdW2)

    TEST_mlp4_dLdb2 = np.allclose(dLdb2.round(4), dLdb2_solution)
    print("Test dLdb2: ", TEST_mlp4_dLdb2)

    TEST_mlp4_dA4dZ3 = np.allclose(dA4dZ3.round(4), dA4dZ3_solution)
    print("\nTest dA4dZ3:", TEST_mlp4_dA4dZ3)

    TEST_mlp4_dLdZ3 = np.allclose(dLdZ3.round(4), dLdZ3_solution)
    print("Test dLdZ3: ", TEST_mlp4_dLdZ3)

    TEST_mlp4_dLdA3 = np.allclose(dLdA3.round(4), dLdA3_solution)
    print("Test dLdA3: ", TEST_mlp4_dLdA3)

    TEST_mlp4_dLdW3 = np.allclose(dLdW3.round(4), dLdW3_solution)
    print("Test dLdW3: ", TEST_mlp4_dLdW3)

    TEST_mlp4_dLdb3 = np.allclose(dLdb3.round(4), dLdb3_solution)
    print("Test dLdb3: ", TEST_mlp4_dLdb3)

    TEST_mlp4_dA5dZ4 = np.allclose(dA5dZ4.round(4), dA5dZ4_solution)
    print("\nTest dA5dZ4:", TEST_mlp4_dA5dZ4)

    TEST_mlp4_dLdZ4 = np.allclose(dLdZ4.round(4), dLdZ4_solution)
    print("Test dLdZ4: ", TEST_mlp4_dLdZ4)

    TEST_mlp4_dLdA4 = np.allclose(dLdA4.round(4), dLdA4_solution)
    print("Test dLdA4: ", TEST_mlp4_dLdA4)

    TEST_mlp4_dLdW4 = np.allclose(dLdW4.round(4), dLdW4_solution)
    print("Test dLdW4: ", TEST_mlp4_dLdW4)

    TEST_mlp4_dLdb4 = np.allclose(dLdb4.round(4), dLdb4_solution)
    print("Test dLdb4: ", TEST_mlp4_dLdb4)

if __name__ == "__main__":
    main()
