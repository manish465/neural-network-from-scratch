import numpy as np


class SigmoidNeuron:
    def __init__(self, input_size, lr=0.1):
        self.weight = np.random.randn(input_size)
        self.bias = 0.0
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        z = np.dot(x, self.weight) + self.bias
        return self.sigmoid(z)

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0

            for xi, target in zip(X, y):
                z = np.dot(xi, self.weight) + self.bias
                pred = self.sigmoid(z)
                loss = 0.5 * (target - pred) ** 2
                total_loss += loss

                dL_dpred = pred - target
                dpred_dz = pred - (1 - pred)

                dz_dw = xi
                dz_db = 1

                self.weight -= self.lr * dL_dpred * dpred_dz * dz_dw
                self.bias -= self.lr * dL_dpred * dpred_dz * dz_db

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss={total_loss:.4f}")
