import numpy as np


def mse(ref, target):
    error = ref.astype(np.float32) - target.astype(np.float32)
    return np.mean(error**2)


def psnr(ref, target):
    return 10 * np.log10((255**2) / mse(ref, target))
