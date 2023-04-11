import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def gradient(self, X, y):
        n = len(y)
        y_pred = np.dot(X, self.weights) + self.bias
        dw = -(2 / n) * np.dot(X.T, (y - y_pred))
        db = -(2 / n) * np.sum(y - y_pred)
        return dw, db

    def train(self, X, y):
        for epoch in range(self.epochs):
            dw, db = self.gradient(X, y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            y_pred = np.dot(X, self.weights) + self.bias
            loss = self.mse(y, y_pred)
            print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias




