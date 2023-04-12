import numpy as np

# Activation func
#-------------------------------------------------------------------------
def sigma(x):
    return 1 / (1 + np.exp(-x))


def g_tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


# Loss func
#-------------------------------------------------------------------------

def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean(axis=-1)


def mean_absolute_error(y_true, y_pred):
    return np.abs(y_true - y_pred).mean(axis=-1)


def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=-1)


def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))


def hinge_loss(y_true, y_pred):
    return np.mean(np.maximum(0, 1 - y_true * y_pred), axis=-1)

# Optimiser
#-------------------------------------------------------------------------
def gradient(weights, bias, X, y):
    n = y.shape[-1]
    y_pred = np.dot(weights, X) + bias
    dw = -(2 / n) * np.dot(X.T, (y - y_pred))
    db = -(2 / n) * np.sum(y - y_pred)
    return dw, db