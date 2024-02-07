import csv
import itertools

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from models.mlp_model import MLP
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import accuracy_score
from utilities import mean_square_error_loss


def train(
    model, X_train, y_train, X_valid, y_valid, lr=0.01, epochs=100, batch_size=32
):
    """
    Trains the MLP model using mini-batch gradient descent.

    Args:
    model (MLP): An instance of the MLP model.
    X_train (ndarray): Training data features.
    y_train (ndarray): Training data labels.
    X_valid (ndarray): Validation data features.
    y_valid (ndarray): Validation data labels.
    lr (float): Learning rate.
    epochs (int): Number of epochs to train the model.
    batch_size (int): Size of the mini-batch for gradient descent.

    Returns:
    tuple: A tuple containing lists of training losses, training accuracies,
           validation losses, and validation accuracies for each epoch.
    """
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        # Shuffle the training data
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        # Mini-batch gradient descent
        for i in range(0, X_train.shape[0], batch_size):
            X_train_batch = X_train_shuffled[i : i + batch_size]
            y_train_batch = y_train_shuffled[i : i + batch_size]

            # Forward Pass
            output = model.forward(X_train_batch)

            # Backward Pass (backpropagation)
            loss = mean_square_error_loss(y_train_batch, output)
            loss_derivative = 2 * (output - y_train_batch) / output.shape[0]
            dW1, dW2 = model.backward(X_train_batch, loss_derivative)

            # Update Weights
            model.W1 -= lr * dW1
            model.W2 -= lr * dW2

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

        # Evaluate training performance
        train_output = model.forward(X_train)
        train_pred_labels = (train_output > 0.5).astype(int)
        train_loss = mean_square_error_loss(y_train, train_output)
        train_accuracy = accuracy_score(y_train.squeeze(), train_pred_labels)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate validation performance
        val_output = model.forward(X_valid)
        val_pred_labels = (val_output > 0.5).astype(int)
        val_loss = mean_square_error_loss(y_valid, val_output)
        val_accuracy = accuracy_score(y_valid.squeeze(), val_pred_labels)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    return train_losses, train_accuracies, val_losses, val_accuracies


def perform_hyperparameter_search(
    hidden_layer_sizes,
    batch_sizes,
    learning_rates,
    epoch_values,
    gamma_values,
    X_train_original,
    y_train,
    X_valid_original,
    y_valid,
    csv_filename,
    dataset,
):
    """
    Performs a grid search over specified hyperparameters for an MLP model,
    including the gamma parameter for RBF feature mapping.
    """
    sns.set(style="whitegrid")

    for gamma, k, batch_size, lr, epochs in itertools.product(
        gamma_values, hidden_layer_sizes, batch_sizes, learning_rates, epoch_values
    ):
        # Transform the data for the current gamma value
        rbf_sampler = RBFSampler(gamma=gamma, n_components=3, random_state=1234)
        X_train = rbf_sampler.fit_transform(X_train_original)
        X_valid = rbf_sampler.transform(X_valid_original)

        # Initialize the model
        model = MLP(input_size=X_train.shape[1], hidden_size=k)

        # Train the model
        train_losses, train_accuracies, val_losses, val_accuracies = train(
            model,
            X_train,
            y_train,
            X_valid,
            y_valid,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
        )

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.lineplot(data=train_losses, label="Train Loss")
        sns.lineplot(data=val_losses, label="Validation Loss")
        plt.title(
            f"Losses for Gamma={gamma}, k={k}, Batch={batch_size}, LR={lr}, Epochs={epochs}"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.subplot(1, 2, 2)
        sns.lineplot(data=train_accuracies, label="Train Accuracy")
        sns.lineplot(data=val_accuracies, label="Validation Accuracy")
        plt.title(
            f"Accuracies for Gamma={gamma}, k={k}, Batch={batch_size}, LR={lr}, Epochs={epochs}"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

        plt.savefig(
            f"images/{dataset}/rbf_gamma_{gamma}_k_{k}_batch_{batch_size}_lr_{lr}_epochs_{epochs}.png"
        )
        plt.close()

        with open(csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    gamma,
                    k,
                    batch_size,
                    lr,
                    epochs,
                    train_losses[-1],
                    val_losses[-1],
                    train_accuracies[-1],
                    val_accuracies[-1],
                ]
            )
