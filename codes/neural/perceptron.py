import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



class Perceptron:
    def __init__(self, input_size, num_classes, learning_rate=0.1, epochs=100):
        self.weights = np.random.randn(num_classes, input_size)
        self.biases = np.random.randn(num_classes)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return (x >= 0).astype(int)

    def predict(self, X):
        z = np.dot(X, self.weights.T) + self.biases
        activations = self.activation_function(z)
        return np.argmax(activations, axis=1)

    def train(self, X, y):
        num_examples = X.shape[0]
        for epoch in range(self.epochs):
            for i in range(num_examples):
                z = np.dot(X[i], self.weights.T) + self.biases
                y_pred = self.activation_function(z)

                y_real = np.zeros_like(self.biases)
                y_real[y[i]] = 1
                delta = self.learning_rate * (y_real - y_pred)
                self.weights += np.outer(delta, X[i])
                self.biases += delta



# Среднеквадратичная ошибка
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Градиент функции потерь
def gradient(X, y, w, b):
    n = len(y)
    y_pred = np.dot(X, w) + b
    dw = -(2 / n) * np.dot(X.T, (y - y_pred))
    db = -(2 / n) * np.sum(y - y_pred)
    return dw, db

# Градиентный спуск
def gradient_descent(X, y, learning_rate=0.01, epochs=100):
    w = np.zeros(X.shape[1])
    b = 0


    for epoch in range(epochs):
        # Вычисление градиента
        dw, db = gradient(X, y, w, b)

        # Обновление параметров
        w -= learning_rate * dw
        b -= learning_rate * db

        # Вычисление и вывод функции потерь
        y_pred = np.dot(X, w) + b
        loss = mse(y, y_pred)

        print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

    return w, b


X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

w, b = gradient_descent(X, y, learning_rate=0.01, epochs=100)
print(f"Optimized weights: {w}")
print(f"Optimized bias: {b}")

