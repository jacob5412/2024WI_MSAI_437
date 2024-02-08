import torch
import torch.nn as nn


class MLPReg(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size1,
        hidden_size2,
        output_size,
    ):
        super(MLPReg, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.tanh = nn.Tanh()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, hidden_size2)
        self.output_layer = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out = self.tanh(self.layer1(x))
        out = self.tanh(self.layer2(out))
        out = self.tanh(self.layer3(out))
        out = torch.sigmoid(self.output_layer(out))
        return out

    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = outputs > 0.5
            return predictions.float()
