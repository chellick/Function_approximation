from model import *
import numpy as np

x_test = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

y_test = np.array([[0], [1], [1], [0]])
    
m = Model(x_test, y_test)

input_layer = Layer((0, 2))
hidden_layer = Layer((2, 2))
output_layer = Layer((2, 1))

m.layers.append(input_layer)
m.layers.append(hidden_layer)
m.layers.append(output_layer)


print(m.forward(m.x_input[0]))

def count_loss(y_true, y_pred):
    


# for l in m.layers:
#     print(l.weights)
