import numpy as np
from base import *


class Layer:
    def __init__(self, shape):
        self.input = None
        self.output = None
        self.shape = shape    #n-previous-neurons, n-neurons
        self.bias = self.create_bias()
        self.weights = self.create_weights()
        
        
        
    def create_weights(self):
        if self.shape[0] == 0:
            return None
        return np.random.randn(self.shape[0], self.shape[1])
        
        
    def create_bias(self):
        return np.random.randn(1, self.shape[1])
    
    
    

    
class Model:
    
    def __init__(self, x_input, y_input, learing_rate=0.1, epochs=100):
        self.x_input = x_input
        self.y_input = y_input
        self.learning_rate = learing_rate
        self.epochs = epochs
        self.layers = []
        
    def forward(self, input):
        sample = input
        self.layers[0].output = sample
        
        for i in range(1, len(self.layers) - 1):
            self.layers[i].input = self.layers[i - 1].output
            self.layers[i].output = np.dot(sample, self.layers[i].weights) + self.layers[i].bias
            sample = sigma(self.layers[i].output)
            
        self.layers[-1].input = sample 
        self.layers[-1].output = np.dot(sample, self.layers[-1].weights) + self.layers[-1].bias
        sample = self.layers[-1].output
        
        return sample
    
    
    # def backward(self, error):


