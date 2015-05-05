import numpy as np

def gaussianFilter(sigma, sample=7.0/2.0):
    n = 2 * round(sample * sigma) + 1
    start = 1
    mid = np.ceil(n / 2.)
    end = n
    x = np.linspace(start - mid, end - mid, n)

    g = np.exp(-(x ** 2) / (2 * (sigma ** 2))) / (sigma * np.sqrt(2 * np.pi))
    return g
