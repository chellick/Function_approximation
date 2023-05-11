from typing import Any
import numpy as np
from models import *
import tensorflow as tf


class Layer:
    def __init__(self, input_shape, weights=None, biases=None):
        self.input_shape = input_shape
        self.weights = self.init_weights() if weights is None else weights
        self.biases = self.init_biases() if biases is None else biases

    def init_weights(self):
       return np.random.rand(self.input_shape[0], self.input_shape[1])
    
    def init_biases(self):
        return np.random.rand(self.input_shape[0], self.input_shape[1])
       
    def __getattributes__(self):
        return self.__dict__

l = Layer((10,2))

print(l.weights)
# print(l.__getattributes__())


class model:
    def __init__(self, input, n_shape, weights=None, biases=None,
                  loss=Loss_function.mean_squared_error,
                  optimizer=Optimisers.gradient):
        
        self.n_shape = n_shape 
        self.input = input
        self.i_shape = input.shape
        self.loss = loss
        self.optimizer = optimizer
        # if weights == None:


    
