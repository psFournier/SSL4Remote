import numpy as np


def minmax(image, m, M):
    
    return np.clip((image - np.reshape(m, (-1, 1, 1))) / np.reshape(M - m, (-1, 1, 1)), 0, 1)
