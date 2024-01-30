import numpy as np
from utilities import (
    initialize_weights,
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
)


class MLP:
    def __init__(self, input_size, hidden_size, seed=42):
        np.random.seed(seed)
        self.W1 = initialize_weights(input_size + 1, hidden_size)
        self.W2 = initialize_weights(hidden_size + 1, 1)

    def forward(self, X):
        X_bias = np.insert(X, 0, 1, axis=1)
        self.Z1 = np.dot(X_bias, self.W1)
        self.A1 = tanh(self.Z1)
        A1_bias = np.insert(self.A1, 0, 1, axis=1)
        self.Z2 = np.dot(A1_bias, self.W2)
        output = sigmoid(self.Z2)
        return output

    def backward(self, X, error, model):
        dZ2 = error * sigmoid_derivative(model.Z2)
        dW2 = np.dot(model.A1.T, dZ2)
        dZ1 = np.dot(dZ2, model.W2[1:].T) * tanh_derivative(model.Z1)
        dW1 = np.dot(X.T, dZ1)
        return dW1, dW2
