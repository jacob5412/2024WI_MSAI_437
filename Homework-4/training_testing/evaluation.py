import torch

from utils.batchifier import get_batch
from utils.modeler import repackage_hidden


def evaluate(model, data_source, criterion, batch_size, bptt):
    model.eval()
    total_loss = 0.0
    hidden = model.init_hidden(batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            loss = criterion(output.view(-1, model.ntokens), targets)
            total_loss += len(data) * loss.item()
    return total_loss / (len(data_source) - 1)
