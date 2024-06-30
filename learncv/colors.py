import numpy as np


def generate(colors_number, seed=None):
    rs = np.random.RandomState(seed)
    return [list(map(int, rs.choice(range(256), size=3))) for _ in range(colors_number)]
