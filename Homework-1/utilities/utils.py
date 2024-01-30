import numpy as np


def initialize_weights(input_size, output_size):
    return np.random.randn(input_size, output_size) * 0.01


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
