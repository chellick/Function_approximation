import numpy as np

# Activation func
#-------------------------------------------------------------------------
class Activation_function:
    @staticmethod
    def sigma(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def g_tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigma_derivative(x):
        sigma_x = Activation_function.sigma(x)
        return sigma_x * (1 - sigma_x)

    @staticmethod
    def g_tanh_derivative(x):
        tanh_x = Activation_function.g_tanh(x)
        return 1 - tanh_x**2

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)




# Loss func
#-------------------------------------------------------------------------

class Loss_function:

    
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean(axis=-1)


    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        return np.abs(y_true - y_pred).mean(axis=-1)


    @staticmethod
    def binary_crossentropy(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=-1)


    @staticmethod
    def categorical_crossentropy(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))


    @staticmethod
    def hinge_loss(y_true, y_pred):
        return np.mean(np.maximum(0, 1 - y_true * y_pred), axis=-1)

# Optimiser
#-------------------------------------------------------------------------

class Optimisers:
    
    @staticmethod
    def gradient(weights, bias, X, y):
        n = y.shape[-1]
        y_pred = np.dot(weights, X) + bias
        dw = -(2 / n) * np.dot(X.T, (y - y_pred))
        db = -(2 / n) * np.sum(y - y_pred)
        return dw, db