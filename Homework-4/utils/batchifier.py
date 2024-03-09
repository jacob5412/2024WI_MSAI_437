def batchify(data, batch_size):
    """
    Converts the data into batches of specified size.

    Args:
        data (Tensor): The input data tensor to be batchified.
        batch_size (int): The number of batches.

    Returns:
        Tensor: A new tensor where each column represents one batch of data,
        and each row within a column represents a single batch element.
    """
    # Calculate the number of full batches that can be made
    nbatch = data.size(0) // batch_size

    # Trim 'data' to make it contain only a complete number of batches.
    # This step ensures that all batches have the same number of elements.
    data = data.narrow(0, 0, nbatch * batch_size)

    # Each column represents one batch of data.
    data = data.view(batch_size, -1).t().contiguous()
    return data


def get_batch(source_data, i, bptt):
    """
    Extracts a batch of data from the source tensor, taking into account
    the backpropagation through time (bptt) length.

    Args:
        source_data (Tensor): The source data from which to extract batches.
        i (int): The index at which to start the batch.
        bptt (int): The backpropagation through time length or sequence length.
    """
    # Determine the effective sequence length for this batch
    seq_len = min(bptt, len(source_data) - 1 - i)
    input_sequence = source_data[i : (i + seq_len)]

    # Extract the target sequence, which is offset by one timestep
    # from the input sequence. This means the target for each timestep
    # in the input sequence is the next token in the source data.
    target_sequence = source_data[(i + 1) : (i + 1 + seq_len)].view(-1)

    return input_sequence, target_sequence
