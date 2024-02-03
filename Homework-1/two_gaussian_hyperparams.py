import csv
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

from models.mlp_model import MLP
from training_testing.training import train
from utilities import load_data

X_train, y_train = load_data("data/two_gaussians_train.csv")
X_valid, y_valid = load_data("data/two_gaussians_valid.csv")
X_test, y_test = load_data("data/two_gaussians_test.csv")
y_train = y_train.reshape(-1, 1)
y_valid = y_valid.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

hidden_layer_sizes = [5, 10, 15, 20, 25, 30]
batch_sizes = [16, 32, 64]
learning_rates = [0.001, 0.01, 0.1]

csv_filename = "results/two_gaussians/hyperparameter_results.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Hidden Layers",
            "Batch Size",
            "Learning Rate",
            "Last Train Loss",
            "Last Validation Loss",
            "Last Train Accuracy",
            "Last Validation Accuracy",
        ]
    )

sns.set(style="whitegrid")

for k, batch_size, lr in itertools.product(
    hidden_layer_sizes, batch_sizes, learning_rates
):
    model = MLP(input_size=X_train.shape[1], hidden_size=k)
    train_losses, train_accuracies, val_losses, val_accuracies = train(
        model,
        X_train,
        y_train,
        X_valid,
        y_valid,
        lr=lr,
        epochs=100,
        batch_size=batch_size,
    )

    # Plot training and validation losses
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.lineplot(data=train_losses, label="Train Loss")
    sns.lineplot(data=val_losses, label="Validation Loss")
    plt.title(f"Losses for k={k}, Batch={batch_size}, LR={lr}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    sns.lineplot(data=train_accuracies, label="Train Accuracy")
    sns.lineplot(data=val_accuracies, label="Validation Accuracy")
    plt.title(f"Accuracies for k={k}, Batch={batch_size}, LR={lr}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    # Save the figure
    filename = f"images/two_gaussians/k_{k}_batch_{batch_size}_lr_{lr}.png"
    plt.savefig(filename)
    plt.close()

    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                k,
                batch_size,
                lr,
                train_losses[-1],
                val_losses[-1],
                train_accuracies[-1],
                val_accuracies[-1],
            ]
        )
