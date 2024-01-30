import numpy as np


def train(model, X, y, lr=0.01, epochs=100):
    for epoch in range(epochs):
        # Forward Pass
        output = model.forward(X)

        # Compute Error
        error = y - output

        # Backward Pass (backpropagation)
        dW1, dW2 = model.backward(X, error, model)

        # Update Weights
        model.W1 += lr * dW1
        model.W2 += lr * dW2
