import numpy as np
import math

class Linear():

    def __init__(self, in_feature, out_feature):
        self.W = np.zeros((in_feature, out_feature))
        self.b = np.zeros((1, out_feature))
        self.feature = None
        self.batch_size = None
        self.ones = None
        self.dLdW = np.zeros((in_feature, out_feature))
        self.dLdb = np.zeros((1, out_feature))
        self.dLdA = None
        self.dZdA = None
        self.dZdW = None
        self.dZdb = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.batch_size = x.shape[0]
        self.feature = x
        self.ones = np.ones((self.batch_size, 1))
        Z = np.dot(self.feature, self.W.T) + (np.dot(self.ones, self.b.T))
        return Z

    def backward(self, delta):
        self.dZdA = self.W.T
        self.dZdW = self.feature
        self.dZdb = self.ones

        self.dLdW= np.dot(delta.T, self.feature)
        self.dLdb= np.mean(delta, axis=0, keepdims=True).T
        self.dLdW = self.dLdW / self.batch_size
        self.dLdA= np.dot(delta, self.dZdA.T)
        return self.dLdA



def main():
    print("──────────────────────────────────────────")
    print("LINEAR | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    A = np.array([
        [-4., -3.],
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")

    W = np.array([
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")

    b = np.array([
        [-1.],
        [0.],
        [1.]], dtype="f")

    linear = Linear(2, 3)
    linear.W = W
    linear.b = b

    Z = linear.forward(A)
    print("Z =\n", Z.round(4), sep="")

    dLdZ = np.array([
        [-4., -3., -2.],
        [-1., -0., 1.],
        [2., 3., 4.],
        [5., 6., 7.]], dtype="f")

    dLdA = linear.backward(dLdZ)

    dZdA = linear.dZdA
    print("\ndZdA =\n", dZdA, sep="")

    dZdW = linear.dZdW
    print("\ndZdW =\n", dZdW, sep="")

    dZdb = linear.dZdb
    print("\ndZdb =\n", dZdb, sep="")

    dLdA = linear.dLdA
    print("\ndLdA =\n", dLdA, sep="")

    dLdA = linear.dLdA
    print("\ndLdA =\n", dLdA, sep="")

    print("\n──────────────────────────────────────────")
    print("LINEAR | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    Z_solution = np.array([
        [10., -3., -16.],
        [4., -1., -6.],
        [-2., 1., 4.],
        [-8., 3., 14.]], dtype="f")

    dZdA_solution = np.array([
        [-2., 0., 2.],
        [-1., 1., 3.]], dtype="f")

    dZdW_solution = np.array([
        [-4., -3.],
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")

    dZdb_solution = np.array([
        [1.],
        [1.],
        [1.],
        [1.]], dtype="f")

    dLdA_solution = np.array([
        [4., -5.],
        [4., 4.],
        [4., 13.],
        [4., 22.]], dtype="f")

    dLdA_solution = np.array([
        [4., -5.],
        [4., 4.],
        [4., 13.],
        [4., 22.]], dtype="f")

    print("\ndZdA =\n", dZdA_solution, sep="")
    print("\ndZdW =\n", dZdW_solution, sep="")
    print("\ndZdb =\n", dZdb_solution, sep="")
    print("\ndLdA =\n", dLdA_solution, sep="")
    print("\ndLdA =\n", dLdA_solution, sep="")

    print("\n──────────────────────────────────────────")
    print("LINEAR | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("\n           Pass?")

    TEST_linear_Z = np.allclose(Z.round(4), Z_solution)
    print("Test Z:   ", TEST_linear_Z)

    TEST_linear_dZdA = np.allclose(dZdA.round(4), dZdA_solution)
    print("Test dZdA:", TEST_linear_dZdA)

    TEST_linear_dZdW = np.allclose(dZdW.round(4), dZdW_solution)
    print("Test dZdW:", TEST_linear_dZdW)

    TEST_linear_dZdb = np.allclose(dZdb.round(4), dZdb_solution)
    print("Test dZdb:", TEST_linear_dZdb)

    TEST_linear_dLdA = np.allclose(dLdA.round(4), dLdA_solution)
    print("Test dLdA:", TEST_linear_dLdA)

    TEST_linear_dLdA = np.allclose(dLdA.round(4), dLdA_solution)
    print("Test dLdA:", TEST_linear_dLdA)

if __name__ == "__main__":
    main()