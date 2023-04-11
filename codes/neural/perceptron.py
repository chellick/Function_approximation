import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import numpy as np

import numpy as np


def sigma(x):
    return 1 / (1 + np.exp(-x))


def g_tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


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


def gradient(weights, bias, X, y):
    n = len(y)
    y_pred = np.dot(X, weights) + bias
    dw = -(2 / n) * np.dot(X.T, (y - y_pred))
    db = -(2 / n) * np.sum(y - y_pred)
    return dw, db


class Neuron:
    def __init__(self, X, y, activation=relu, loss=mean_squared_error,
                 learning_rate=0.01, num_iterations=1000, optimizer=gradient):
        self.X = X
        self.y = y
        self.weights = np.random.randn(X.shape[1])
        self.activation = activation
        self.loss = loss
        self.bias = 0
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.optimizer = optimizer

    def activation_function(self):
        weighted_sum = np.dot(self.X, self.weights) + self.bias
        return self.activation(weighted_sum)

    def train(self):
        for epoch in range(self.num_iterations):
            dw, db = self.optimizer(self.weights, self.bias, self.X, self.y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            y_pred = self.activation_function()
            loss = self.loss(self.y, y_pred)
            print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

    def predict(self, X):
        weighted_sum = np.dot(X, self.weights) + self.bias
        return self.activation(weighted_sum)


