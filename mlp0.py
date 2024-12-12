from loss import *
from activation import *
from batchnorm import *
from linear import *

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

if __name__ == "__main__":
    main()
