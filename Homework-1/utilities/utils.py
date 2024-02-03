import numpy as np


def initialize_weights(input_size, output_size):
    return np.random.randn(input_size, output_size) * 0.01


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def tanh(x):
    return 2 * sigmoid(2 * x) - 1


def tanh_derivative(x):
    tanh_x = tanh(x)
    return 1 - tanh_x**2


def sigmoid_derivative(x):
    sig_x = sigmoid(x)
    return sig_x * (1 - sig_x)


def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / len(
        y_true
    )
    return loss
