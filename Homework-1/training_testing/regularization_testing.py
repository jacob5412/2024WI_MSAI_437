import torch
def evaluate_model_classification(model, loader, criterion):
    model.eval()  # Evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += inputs.size(0)
    
    mean_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return mean_loss, accuracy

def train_and_evaluate_model(model, criterion, optimizer, train_loader, valid_loader, test_loader, num_epochs, l1_strength=0.0):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
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
    
    train_loss, train_accuracy = evaluate_model_classification(model, train_loader, criterion)
    valid_loss, valid_accuracy = evaluate_model_classification(model, valid_loader, criterion)
    test_loss, test_accuracy = evaluate_model_classification(model, test_loader, criterion)

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
