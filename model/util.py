import numpy as np


def softmax(x, temperature=1.0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x))/temperature)
    return e_x / e_x.sum(axis=0)  # only difference
