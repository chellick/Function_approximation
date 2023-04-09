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



