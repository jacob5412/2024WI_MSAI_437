import torch
import torch.nn as nn
import torch.optim as optim


def train_mtautoencoder(
    model, train_loader, val_loader, num_epochs, lr, weight_decay, lambda_classification
):
    criterion_mse = nn.MSELoss()
    criterion_classification = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_mse_losses = []
    train_classification_losses = []
    train_accuracies = []
    val_mse_losses = []
    val_classification_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training Loop
        model.train()
        running_train_mse_loss = 0.0
        running_train_classification_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            images, labels = batch["image"], batch["class"]
            optimizer.zero_grad()

            reconstructed, classification_logits = model(images)
            loss_mse = criterion_mse(reconstructed, images)
            loss_classification = criterion_classification(
                classification_logits, labels
            )
            total_loss = loss_mse + lambda_classification * loss_classification
            total_loss.backward()
            optimizer.step()

            running_train_mse_loss += loss_mse.item() * images.size(0)
            running_train_classification_loss += (
                loss_classification.item() * labels.size(0)
            )

            _, predicted = torch.max(classification_logits.data, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_mse_loss = running_train_mse_loss / len(train_loader.dataset)
        epoch_train_classification_loss = running_train_classification_loss / len(
            train_loader.dataset
        )
        epoch_train_accuracy = 100 * correct_train / total_train
        train_mse_losses.append(epoch_train_mse_loss)
        train_classification_losses.append(epoch_train_classification_loss)
        train_accuracies.append(epoch_train_accuracy)

        # Validation loop
        model.eval()
        running_val_mse_loss = 0.0
        running_val_classification_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch["image"], batch["class"]
                reconstructed, classification_logits = model(images)
                loss_mse = criterion_mse(reconstructed, images)
                loss_classification = criterion_classification(
                    classification_logits, labels
                )

                running_val_mse_loss += loss_mse.item() * images.size(0)
                running_val_classification_loss += (
                    loss_classification.item() * labels.size(0)
                )

                _, predicted = torch.max(classification_logits.data, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_mse_loss = running_val_mse_loss / len(val_loader.dataset)
        epoch_val_classification_loss = running_val_classification_loss / len(
            val_loader.dataset
        )
        epoch_val_accuracy = 100 * correct_val / total_val
        val_mse_losses.append(epoch_val_mse_loss)
        val_classification_losses.append(epoch_val_classification_loss)
        val_accuracies.append(epoch_val_accuracy)

        if epoch % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Train MSE Loss: {epoch_train_mse_loss:.4f}, "
                + f"Train Classification Loss: {epoch_train_classification_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2f}%, "
                + f"Val MSE Loss: {epoch_val_mse_loss:.4f}, Val Classification Loss: {epoch_val_classification_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.2f}%"
            )

    return (
        model,
        train_mse_losses,
        train_classification_losses,
        train_accuracies,
        val_mse_losses,
        val_classification_losses,
        val_accuracies,
    )
