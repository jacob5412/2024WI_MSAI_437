import numpy as np

from utilities import binary_cross_entropy_loss


def train(
    model, X_train, y_train, X_valid, y_valid, lr=0.01, epochs=100, batch_size=32
):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        for i in range(0, X_train.shape[0], batch_size):
            # Mini-batch data
            X_train_batch = X_train_shuffled[i : i + batch_size]
            y_train_batch = y_train_shuffled[i : i + batch_size]

            # Forward Pass
            output = model.forward(X_train_batch)

            # Backward Pass (backpropagation)
            loss = binary_cross_entropy_loss(y_train_batch, output)
            loss_derivative = output - y_train_batch
            dW1, dW2 = model.backward(X_train_batch, loss_derivative)

            # Update Weights
            model.W1 -= lr * dW1
            model.W2 -= lr * dW2

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

        # Record training loss and accuracy
        train_pred = model.predict(X_train)
        train_loss = binary_cross_entropy_loss(y_train, train_pred)
        train_accuracy = np.mean(train_pred == y_train.squeeze())
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation step
        val_pred = model.predict(X_valid)
        val_loss = binary_cross_entropy_loss(y_valid, val_pred)
        val_accuracy = np.mean(val_pred == y_valid.squeeze())
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    return train_losses, train_accuracies, val_losses, val_accuracies
