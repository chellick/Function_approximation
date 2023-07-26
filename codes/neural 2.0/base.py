import numpy as np


def sigma(x):
    return 1/(1 + np.exp(-x))

def sigma_d(x):
    return sigma(x) * (1 - sigma(x))

def relu(x):
    return max(0, x)

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_d(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_pred.size

