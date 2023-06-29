import numpy as np
from base import Activation_function, Optimisers, Loss_function


class Layer:
    def __init__(self, shape=tuple):
        self.shape = shape # first stands for input shape: n-neurons of prev layer, second: n-neurons of given layer
        self.weights = self.init_weights()
        self.biases = self.init_biases()
        self.input = None
        self.output = None

    
    def init_weights(self):
        return np.random.randn(self.shape[0], self.shape[1])

    def init_biases(self):
        return np.random.randn(1, self.shape[1])

class Model:
    def __init__(self, x_input, y_input, activation, optimiser, loss, 
                 typen=None, shape=(tuple), learning_rate=0.1, epochs=1000):
        self.x_input = x_input
        self.y_input = y_input
        self.shape = shape  # for example shape (10, 4) will direct to create 10 layer model with 4 neurons in each hidden layer
        self.activation = activation
        self.optimiser = optimiser
        self.loss = loss
        self.layers = []
        self.typen = typen
        self.leatning_rate = learning_rate
        self.epochs = epochs

    

    def create_base(self, first_shape=None):
        if self.typen == None or self.typen == 'Single':
            if first_shape is None:
                start = Layer((1, self.shape[1])) # type: ignore # start shape with 1 neuron input 
            else:
                start = Layer((0, first_shape))
            self.layers.append(start)
            
            for _ in range(self.shape[0]):
                layer = Layer((self.shape[1], self.shape[1]))
                self.layers.append(layer)
                

            final = Layer((self.shape[1], 1))
            self.layers.append(final)

        return 'Base created'
    
    def forward(self, x_input):
        output = x_input
        for layer in self.layers:
            layer.input = output
            layer.output = np.dot(output, layer.weights) + layer.biases
            output = self.activation(layer.output)
            
        return output
    
    def backward(self, error):
        for layer in reversed(self.layers):
            error = error * Activation_function.sigma_derivative(layer.output) # TODO make derivatives
            layer.biases -= self.leatning_rate * error
            layer.weights -= self.leatning_rate * error
            error = np.dot(error, layer.weights.T)
            

    def fit(self):
        err = 0
        for epoch in range(self.epochs):
            for i in range(len(self.x_input)):
                output = self.forward(self.x_input[i])
                error = Loss_function.mse_derivative(self.y_input[1], output) # TODO make derivatives
                self.backward(error)
                err = Loss_function.mean_squared_error(self.y_input[i], output)

            print(f'{err / len(self.x_input):.3f}, error')
            

x_train = np.array(list(i for i in range(100)))
y_train = x_train ** 2

x_train = [float(i)/max(x_train) for i in x_train]
y_train = [float(i)/max(y_train) for i in y_train]


m = Model(x_train, y_train, Activation_function.sigma, 
          Optimisers.gradient, Loss_function.mean_squared_error, 
          None, (4, 2))


m.create_base()
m.fit()






# m.mareuh.THE-receptov




