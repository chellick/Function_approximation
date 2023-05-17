from typing import Any
import numpy as np
from models import *
import tensorflow as tf


class Layer:
    def __init__(self, output_shape, input_shape, weights=None, biases=None):

        self.input = None
        self.output = None
        self.output_shape = output_shape # N neurons
        self.input_shape = input_shape
        self.weights = self.init_weights() if weights is None else weights
        self.biases = self.init_biases() if biases is None else biases

    def init_weights(self):
       return np.random.rand(self.input_shape, self.output_shape)
    
    def init_biases(self):
        return np.random.rand(1, self.output_shape) 
    
    def __getattributes__(self):
        return self.__dict__
    


class model:
    def __init__(self, input_x, input_y, epochs=100, learning_rate=0.01, batch_size=10,
                 loss=Loss_function.mean_squared_error, activation=Activation_function.sigma):
                
        self.input_x = np.array(input_x)
        self.input_y = np.array(input_y)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.loss_d = Loss_function.mse_derivative if loss == Loss_function.mean_squared_error else Loss_function.mse_derivative
        self.activation = activation
        self.activation_d = Activation_function.sigma_derivative if activation == Activation_function.sigma else Activation_function.sigma_derivative
        self.layers = []
        self.batch_size = batch_size

    def add_layer(self, layer):
        self.layers.append(layer)
    

    def forward(self, input):
        self.layers[0].input = input
        self.layers[0].output = np.dot(input, self.layers[0].weights) + self.layers[0].biases

        for i in range(1, len(self.layers)):
            self.layers[i].input = self.activation(self.layers[i - 1].output)
            self.layers[i].output = np.dot(self.layers[i].input, self.layers[i].weights) + self.layers[i].biases

    
    def backward(self, error):
        err = [error]
        
        for layer in reversed(self.layers):
            input_error = self.activation_d(layer.output) * err[-1]
            layer.weights -= layer.input.T * input_error * self.learning_rate
            layer.biases -= input_error * self.learning_rate
            err.append(np.dot(input_error, layer.weights.T))


        return err

    def fit(self):
        for j in range(self.epochs):
            err = 0
            for i in range(len(self.input_x)):
                self.forward(self.input_x[i])
                error = self.loss_d(self.input_y[i], self.layers[-1].output)
                self.backward(error)
                err += self.loss(self.input_y[i], self.layers[-1].output)

            print(f'{(err / len(self.input_x) * 100):.3f}, error')
                # print(self.input_y[i])
