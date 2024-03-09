from collections import Counter

import torch


class Vocabulary:
    """
    A simple vocabulary manager that can map tokens to indices and vice versa.
    It can also numericalize a list of tokens into a list of indices.
    """

    def __init__(self):
        """
        Initializes the Vocabulary instance, setting up the underlying data structures
        for token and index mapping.
        """
        self.token_to_idx = {"<unk>": 0}  # Maps tokens to their indices
        self.idx_to_token = [
            "<unk>"
        ]  # A list of tokens indexed by their numerical indices
        self.size = 1  # The current size of the vocabulary

    def add_token(self, token):
        """
        Adds a token to the vocabulary.

        Args:
            token (str): The token to be added.

        Returns:
            int: The index of the token in the vocabulary.
        """
        if token not in self.token_to_idx:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = self.size
            self.size += 1
        return self.token_to_idx[token]

    def itos(self, index):
        return self.idx_to_token[index] if index < len(self.idx_to_token) else "<unk>"

    def numericalize(self, tokens):
        """
        Converts a list of tokens into their corresponding indices.

        Args:
            tokens (list of str): The tokens to be numericalized.

        Returns:
            torch.Tensor: A tensor of the tokens converted to their numerical indices.
        """
        indices = [
            self.token_to_idx.get(token, self.token_to_idx["<unk>"]) for token in tokens
        ]
        return torch.tensor(indices, dtype=torch.long)

    def build_vocab(self, tokens_list):
        """
        Builds the vocabulary from a list of tokens.

        Args:
            tokens_list (list of str): The tokens used to build the vocabulary.
        """
        for token in tokens_list:
            self.add_token(token)


def process_corpus(
    file_path, vocab_size=10000, replaced_tokens_filepath="data/replaced_tokens.txt"
):
    """
    Processes a text corpus to replace less frequent tokens with '<unk>'.

    Args:
        file_path (str): The path to the corpus file.
        vocab_size (int): The number of tokens to keep in the vocabulary.
        replaced_tokens_filepath (str): The path to save the replaced tokens.

    Returns:
        list of str: The processed tokens list with less frequent tokens replaced by '<unk>'.
    """
    # We're using UTF just to be safe
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    tokens = text.split(" ")

    token_freq = Counter(tokens)

    # Determine the frequency threshold for less frequent tokens
    freq_threshold = sorted(token_freq.values(), reverse=True)[vocab_size]

    # Replace less frequent tokens with <unk> and save replaced tokens
    replaced_tokens = set()
    processed_tokens = []
    for token in tokens:
        if token_freq[token] < freq_threshold:
            processed_tokens.append("<unk>")
            replaced_tokens.add(token)
        else:
            processed_tokens.append(token)

    with open(replaced_tokens_filepath, "w", encoding="utf-8") as file:
        for token in replaced_tokens:
            file.write(token + "\n")

    return processed_tokens


def read_corpus(file_path, replaced_tokens_filepath="data/replaced_tokens.txt"):
    """
    Reads a corpus and replaces specified tokens with '<unk>'.

    Args:
        file_path (str): The path to the corpus file.
        replaced_tokens_filepath (str): The file containing tokens to be replaced by '<unk>'.

    Returns:
        list of str: The processed tokens list with specified tokens replaced by '<unk>'.
    """
    with open(replaced_tokens_filepath, "r", encoding="utf-8") as file:
        replaced_tokens = {line.strip() for line in file}

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    tokens = text.split(" ")

    # Replace tokens found in the replaced tokens list with <unk>
    processed_tokens = []
    for token in tokens:
        if token in replaced_tokens:
            processed_tokens.append("<unk>")
        else:
            processed_tokens.append(token)

    return processed_tokens
