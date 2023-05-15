from perceptron import *
import numpy as np

x_inputs = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
y_inputs = np.array([[0], [0], [1], [0]])
x, y = x_inputs[0].T, y_inputs[0]

l = Layer(3, 2)
l1 = Layer(1, 3)

mod = model(x_inputs, y_inputs)
mod.layers.append(l)
mod.layers.append(l1)

# print(mod.layers[0].output)
# print(mod.layers[1].output) # working alright

mod.forward(mod.layers[0], mod.input_x[0])
mod.forward(mod.layers[1], mod.layers[0].output)

print(mod.layers[0].output)
print(mod.layers[1].output)

# for i in range(len(mod.layers)):
#     print(f"weights in {i} layer: \n {mod.layers[i].weights}")

# mod.backward(mod.input_y[0])



# for i in range(len(mod.layers)):
#     print(f"weights in {i} layer: \n {mod.layers[i].weights}")
