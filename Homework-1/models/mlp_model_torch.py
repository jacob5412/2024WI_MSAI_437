import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Assuming the conversion of datasets to PyTorch tensors and DataLoader setup is done similar to previous examples

# Neural Network Definition
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.0):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        return out