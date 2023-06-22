import numpy as np
from base import Activation_function, Optimisers, Loss_function


class Layer:
    def __init__(self, shape=tuple):
        self.shape = shape # first stands for input shape: n-neurons of prev layer, second: n-neurons of given layer
        self.weights = self.init_weights()
        self.biases = self.init_biases()
    
    def init_weights(self):
        return np.random.randn(self.shape[0], self.shape[1])

    def init_biases(self):
        return np.random.randn(self.shape[0], self.shape[1])

class Model:
    def __init__(self, x_input, y_input, activation, optimiser, loss, typen=None, shape=(tuple)):
        self.x_input = x_input
        self.y_input = y_input
        self.shape = shape  # for example shape (10, 4) will direct to create 10 layer model with 4 neurons in each hidden layer
        self.activation = activation
        self.optimiser = optimiser
        self.loss = loss
        self.layers = []
        self.typen = typen
    

    def create_base(self, first_shape=None):
        if self.typen == None or self.typen == 'Single':
            if first_shape is None:
                start = Layer((0, 1)) # start shape with 1 neuron input 
            else:
                start = Layer((0, first_shape))
            self.layers.append(start)
            
            for _ in range(self.shape[0]):
                layer = Layer((self.shape[1], self.shape[1]))
                self.layers.append(layer)
                

            final = Layer((self.shape[1], 1))
            self.layers.append(final)

        return 'Base created'

m = Model([], [], Activation_function.sigma, 
          Optimisers.gradient, Loss_function.mean_squared_error, 
          None, (4, 2))



m.create_base()

for l in m.layers:
    print(l.weights)
