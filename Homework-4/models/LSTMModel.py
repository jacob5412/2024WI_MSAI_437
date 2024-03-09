import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    A recurrent neural network model using LSTM units with tied weights.

    Attributes:
        ntokens (int): Number of tokens in the vocabulary.
        nhid (int): Number of hidden units in the LSTM.
        encoder (Embedding): Embedding layer to transform tokens into continuous vectors.
        drop (Dropout): Dropout layer to help prevent overfitting.
        lstm (LSTM): LSTM layer that processes the sequence data.
    """

    def __init__(self, ntokens, ninp=100, nhid=100, dropout=0.5):
        """
        Initializes the LSTM model with an embedding layer, a dropout layer,
        and a single-layer LSTM, with weight tying between the embedding and output layer.

        Args:
            ntokens (int): Number of tokens in the vocabulary.
            ninp (int): Dimensionality of the embedding space (number of input features).
            nhid (int): Number of hidden units in the LSTM layer.
            dropout (float): Dropout probability used in the dropout layer.
        """
        super(LSTMModel, self).__init__()
        self.ntokens = ntokens
        self.nhid = nhid
        self.encoder = nn.Embedding(ntokens, ninp)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(ninp, nhid, num_layers=1)

        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of the embedding layer uniformly within a small range.
        """
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        """
        Defines the forward pass of the LSTM model.

        Args:
            input (Tensor): A tensor of input data of shape (seq_len, batch_size).
            hidden (tuple): A tuple containing the initial hidden and cell states of the LSTM.

        Returns:
            Tuple[Tensor, tuple]: A tuple containing the output tensor from the LSTM (of shape
            (seq_len, batch_size, nhid)) and the hidden state tensor for the last time step.
        """
        emb = self.encoder(input)
        emb = self.drop(emb)
        output, hidden = self.lstm(emb, hidden)

        # Apply weight tying between encoder and output projection
        output = torch.matmul(output, self.encoder.weight.transpose(0, 1))

        return output, hidden

    def init_hidden(self, batch_size):
        """
        Initializes a new tensor for the hidden and cell states of the LSTM.

        Args:
            batch_size (int): The size of the batch.

        Returns:
            tuple: A tuple of tensors with zeros with shape (num_layers, batch_size,
            nhid) for the hidden state and cell state.
        """
        return (
            torch.zeros(1, batch_size, self.nhid),
            torch.zeros(1, batch_size, self.nhid),
        )
