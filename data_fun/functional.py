import numpy as np


def rmlse(y_hat,y):
    return np.sqrt( sum((np.log(y_hat + 1) - np.log(y+1))**2) /len(y))

def split_data(x, weight = 0.9):
    """
    split_data into to parts. |part1|/ |part1 + part2| == weight
    """
    offset = int(len(x) * weight)
    return x[:offset], x[offset:]
