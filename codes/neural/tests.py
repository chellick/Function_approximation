from perceptron import *
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x ** 2

x_inputs = np.linspace(0, 11, 1000)
y_inputs = f(x_inputs)

l1 = Layer(16, 1)
l2 = Layer(16, 16)
l3 = Layer(16, 16)
l4 = Layer(1, 16)

mod = model(x_inputs, y_inputs)


mod.add_layer(l1)
mod.add_layer(l2)
mod.add_layer(l3)
mod.add_layer(l4)


mod.forward(mod.input_x[0])

error = Loss_function.mean_squared_error(mod.input_y[0], mod.layers[-1].output)



for i in range(len(mod.layers)):
    print(f"weights in {i} layer: \n {mod.layers[i].weights} \n")


mod.fit()



for i in range(len(mod.layers)):
    print(f"weights in {i} layer: \n {mod.layers[i].weights} \n")


plt.plot(x_inputs, y_inputs, label='train')
plt.plot(x_inputs, [mod.forward(i) for i in mod.input_y], label='test')

plt.show()