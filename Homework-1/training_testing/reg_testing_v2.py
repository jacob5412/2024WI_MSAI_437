import torch
from sklearn.metrics import accuracy_score


def evaluate_model_classification(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            predicted = model.predict(inputs)
            all_predictions.extend(predicted.squeeze().tolist())
            all_targets.extend(targets.tolist())

    mean_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_predictions)
    return mean_loss, accuracy


def train_and_evaluate_model(
    model,
    criterion,
    optimizer,
    train_loader,
    valid_loader,
    test_loader,
    num_epochs,
    l1_strength=0.0,
):
    for _ in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # L1 Regularization
            if l1_strength > 0.0:
                l1_penalty = sum(p.abs().sum() for p in model.parameters())
                loss += l1_strength * l1_penalty

            loss.backward()
            optimizer.step()

    train_loss, train_accuracy = evaluate_model_classification(
        model, train_loader, criterion
    )
    valid_loss, valid_accuracy = evaluate_model_classification(
        model, valid_loader, criterion
    )
    test_loss, test_accuracy = evaluate_model_classification(
        model, test_loader, criterion
    )

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
