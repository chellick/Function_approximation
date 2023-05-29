from perceptron import *
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x ** 2

x_inputs = np.linspace(0, 10, 100)
y_inputs = f(x_inputs)

l1 = Layer(32, 1)
l2 = Layer(32, 32)
l3 = Layer(32, 32)
l4 = Layer(1, 32)

mod = model(x_inputs, y_inputs)


mod.add_layer(l1)
mod.add_layer(l2)
mod.add_layer(l3)
mod.add_layer(l4)




for i in range(len(mod.layers)):
    print(f"weights in {i} layer: \n {mod.layers[i].weights} \n")


mod.fit()



for i in range(len(mod.layers)):
    print(f"weights in {i} layer: \n {mod.layers[i].weights} \n")

