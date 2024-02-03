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

    def backward(self, X, loss_derivative):
        dZ2 = loss_derivative * sigmoid_derivative(self.Z2)
        A1_bias = np.insert(self.A1, 0, 1, axis=1)
        dW2 = np.dot(A1_bias.T, dZ2)

        dZ1 = np.dot(dZ2, self.W2[1:, :].T) * tanh_derivative(self.Z1)
        X_bias = np.insert(X, 0, 1, axis=1)
        dW1 = np.dot(X_bias.T, dZ1)

        return dW1, dW2

    def predict(self, X):
        X_bias = np.insert(X, 0, 1, axis=1)
        Z1 = np.dot(X_bias, self.W1)
        A1 = tanh(Z1)
        A1_bias = np.insert(A1, 0, 1, axis=1)
        Z2 = np.dot(A1_bias, self.W2)
        output = sigmoid(Z2)

        predictions = (output > 0.5).astype(int)
        return predictions.squeeze()
