# Prepare data for PyTorch
import torch
from torch.utils.data import TensorDataset

def format_data(X, y):
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).view(-1, 1)  # Reshape y for PyTorch
    return TensorDataset(X_tensor, y_tensor)