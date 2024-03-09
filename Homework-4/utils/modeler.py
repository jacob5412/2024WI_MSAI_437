import torch


def repackage_hidden(h):
    """
    Detaches hidden states from their history to prevent backpropagating through the entire history.

    Args:
        h (Tensor or tuple of Tensors): The hidden state tensor(s).

    Returns:
        Tensor or tuple of Tensors: The same hidden states detached from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
