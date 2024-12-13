import numpy as np
from linear import *
from activation import *


class SGD:

    def __init__(self, model, lr=0.1, momentum=0):

        self.l = model.layers
        self.L = len(model.layers)
        self.lr = lr
        self.mu = momentum
        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f") for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f") for i in range(self.L)]

    def step(self):

        for i in range(self.L):

            if self.mu == 0:

                self.l[i].W = self.l[i].W - self.lr * self.l[i].dLdW
                self.l[i].b = self.l[i].b - self.lr * self.l[i].dLdb

            else:

                self.v_W[i] = None  # TODO
                self.v_b[i] = None  # TODO
                self.l[i].W = None  # TODO
                self.l[i].b = None  # TODO

def main():
    print("\n──────────────────────────────────────────")
    print("SGD | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    class PseudoModel:
        def __init__(self):
            self.layers = [Linear(3, 2)]
            self.f = [ReLu()]

        def forward(self, A):
            return NotImplemented

        def backward(self):
            return NotImplemented

    # Create Example Model
    pseudo_model = PseudoModel()

    pseudo_model.layers[0].W = np.ones((3, 2))
    pseudo_model.layers[0].dLdW = np.ones((3, 2)) / 10
    pseudo_model.layers[0].b = np.ones((3, 1))
    pseudo_model.layers[0].dLdb = np.ones((3, 1)) / 10

    print("\nInitialized Parameters:\n")
    print("W =\n", pseudo_model.layers[0].W, "\n", sep="")
    print("b =\n", pseudo_model.layers[0].b, "\n", sep="")

    # Test Example Models
    optimizer = SGD(pseudo_model, lr=0.9)
    optimizer.step()

    print("Parameters After SGD (Step=1)\n")

    W_1 = pseudo_model.layers[0].W.copy()
    b_1 = pseudo_model.layers[0].b.copy()
    print("W =\n", W_1, "\n", sep="")
    print("b =\n", b_1, "\n", sep="")

    optimizer.step()

    print("Parameters After SGD (Step=2)\n")

    W_2 = pseudo_model.layers[0].W
    b_2 = pseudo_model.layers[0].b
    print("W =\n", W_2, "\n", sep="")
    print("b =\n", b_2, "\n", sep="")

    print("──────────────────────────────────────────")
    print("SGD | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    W_1_solution = np.array([
        [0.91, 0.91],
        [0.91, 0.91],
        [0.91, 0.91]], dtype="f")

    b_1_solution = np.array([
        [0.91],
        [0.91],
        [0.91]], dtype="f")

    W_2_solution = np.array([
        [0.82, 0.82],
        [0.82, 0.82],
        [0.82, 0.82]], dtype="f")

    b_2_solution = np.array([
        [0.82],
        [0.82],
        [0.82]], dtype="f")

    print("\nParameters After SGD (Step=1)\n")

    print("W =\n", W_1_solution, "\n", sep="")
    print("b =\n", b_1_solution, "\n", sep="")

    print("Parameters After SGD (Step=2)\n")

    print("W =\n", W_2_solution, "\n", sep="")
    print("b =\n", b_2_solution, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("SGD | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("                 Pass?")

    TEST_sgd_W_1 = np.allclose(W_1.round(4), W_1_solution)
    print("Test W (Step 1):", TEST_sgd_W_1)

    TEST_sgd_b_1 = np.allclose(b_1.round(4), b_1_solution)
    print("Test b (Step 1):", TEST_sgd_b_1)

    TEST_sgd_W_2 = np.allclose(W_2.round(4), W_2_solution)
    print("Test W (Step 2):", TEST_sgd_W_2)

    TEST_sgd_b_2 = np.allclose(b_2.round(4), b_2_solution)
    print("Test b (Step 2):", TEST_sgd_b_2)


if __name__ == "__main__":
    main()