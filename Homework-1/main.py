import matplotlib.pyplot as plt

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
best_k = None
best_accuracy = 0

for k in hidden_layer_sizes:
    model = MLP(input_size=X_train.shape[1], hidden_size=k)
    train_losses, train_accuracies, val_losses, val_accuracies = train(
        model, X_train, y_train, X_valid, y_valid, lr=0.01, epochs=100, batch_size=32
    )

    # Plot training and validation losses
    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"Losses for k={k}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.title(f"Accuracies for k={k}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Save the figure
    plt.savefig(f"images/two_gaussians/k_{k}.png")
    plt.close()
