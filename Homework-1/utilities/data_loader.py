import numpy as np


def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    X, y = data[:, 1:], data[:, 0]
    return X, y
