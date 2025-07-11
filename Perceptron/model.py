import numpy as np


class Perceptron:
    def __init__(self, input_size, lr=0.01):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = lr

    def activation(self, x):
        return 1 if x >= 0 else 0  # Step Function

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.activation(z)

    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                # Perceptron update rule
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
