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
    def __init__(self, input_x, input_y, size=3, hidden_size=2, epochs=10, learning_rate=0.1):
        self.input_x = np.array(input_x)
        self.input_y = np.array(input_y)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.size = size
        self.hidden_size = hidden_size
        self.layers = []


    def forward(self, layer, input):
        layer.input = input
        layer.output = Activation_function.g_tanh(np.dot(input, layer.weights) + layer.biases)

        return layer.output
    


    # def backward(self, y_true):
        


    def fit(self):
        samples = self.input_y.size
        # for i in range(self.epochs):
        #     for j in range(len(self.layers)):
                
