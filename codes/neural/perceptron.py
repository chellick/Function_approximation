import numpy as np
from models import *

class Neuron:
    def __init__(self, x_true, y_true, weights=None, biases=None, optimiser=gradient,
                  loss=mean_squared_error, activation=sigma, epochs=1000, learning_rate=0.01, hidden_size=4, input_size=3):

        self.x_true = x_true
        self.y_true = y_true
        self.optimiser = optimiser
        self.loss = loss
        self.activation = activation
        self.epochs = epochs
        self.learning_rate = learning_rate

        # Инициализация весов и смещений для двух скрытых слоев и выходного слоя
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.biases1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, hidden_size)
        self.biases2 = np.zeros((1, hidden_size))
        self.weights3 = np.random.randn(hidden_size, 1)
        self.biases3 = np.zeros((1, 1))


