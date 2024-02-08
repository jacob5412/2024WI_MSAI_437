import csv
import itertools

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from models.mlp_model_reg import MLPReg
from sklearn.metrics import accuracy_score


def train(
    model,
    criterion,
    optimizer,
    X_train_tensor,
    y_train_tensor,
    X_valid_tensor,
    y_valid_tensor,
    epochs,
    batch_size,
):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        # Shuffle the training data
        permutation = torch.randperm(X_train_tensor.size()[0])
        for i in range(0, X_train_tensor.size()[0], batch_size):
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())

        # Calculate accuracy
        with torch.no_grad():
            # Training accuracy
            train_pred = model.predict(X_train_tensor)
            train_acc = accuracy_score(
                y_train_tensor.cpu().numpy(), train_pred.cpu().numpy()
            )
            train_accuracies.append(train_acc)

            # Validation accuracy
            val_pred = model.predict(X_valid_tensor)
            val_acc = accuracy_score(
                y_valid_tensor.cpu().numpy(), val_pred.cpu().numpy()
            )
            val_accuracies.append(val_acc)

            val_loss = criterion(model(X_valid_tensor), y_valid_tensor).item()
            val_losses.append(val_loss)

        if epoch % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return train_losses, train_accuracies, val_losses, val_accuracies


def perform_hyperparameter_search(
    hidden_layer_sizes,
    batch_sizes,
    learning_rates,
    epoch_values,
    weight_decay_rates,
    X_train_tensor,
    y_train_tensor,
    X_valid_tensor,
    y_valid_tensor,
    csv_filename,
    dataset,
):
    sns.set(style="whitegrid")

    for (
        hidden_size1,
        hidden_size2,
        batch_size,
        lr,
        epochs,
        weight_decay,
    ) in itertools.product(
        hidden_layer_sizes,
        hidden_layer_sizes,
        batch_sizes,
        learning_rates,
        epoch_values,
        weight_decay_rates,
    ):
        model = MLPReg(
            input_size=X_train_tensor.shape[1],
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            output_size=1,
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        train_losses, train_accuracies, val_losses, val_accuracies = train(
            model,
            criterion,
            optimizer,
            X_train_tensor,
            y_train_tensor,
            X_valid_tensor,
            y_valid_tensor,
            epochs,
            batch_size,
        )

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.lineplot(data=train_losses, label="Train Loss")
        sns.lineplot(data=val_losses, label="Validation Loss")
        plt.title(
            f"Losses - Hidden1={hidden_size1}, Hidden2={hidden_size2}, Batch={batch_size}, LR={lr}, Weight Decay={weight_decay}, Epochs={epochs}"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.subplot(1, 2, 2)
        sns.lineplot(data=train_accuracies, label="Train Accuracy")
        sns.lineplot(data=val_accuracies, label="Validation Accuracy")
        plt.title(
            f"Accuracies - Hidden1={hidden_size1}, Hidden2={hidden_size2}, Batch={batch_size}, LR={lr}, Weight Decay={weight_decay}, Epochs={epochs}"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

        plt.savefig(
            f"images/{dataset}/reg_l2_weight_decay_{weight_decay}_hidden1_{hidden_size1}_hidden2_{hidden_size2}_batch_{batch_size}_lr_{lr}_epochs_{epochs}.png"
        )
        plt.close()

        with open(csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    hidden_size1,
                    hidden_size2,
                    batch_size,
                    weight_decay,
                    lr,
                    epochs,
                    train_losses[-1],
                    val_losses[-1],
                    train_accuracies[-1],
                    val_accuracies[-1],
                ]
            )
