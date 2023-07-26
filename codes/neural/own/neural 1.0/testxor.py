import numpy as np
from perceptron import *

x = [[0,0],
     [0,1],
     [1,0],
     [1,1]]
y = [[0], [1], [1], [0]]


mod = model(x, y)

l1 = Layer(2, 2)
l2 = Layer(2, 2)
l3 = Layer(2, 2)
l4 = Layer(1, 2)

mod.add_layer(l1)
mod.add_layer(l2)
mod.add_layer(l3)
mod.add_layer(l4)



for i in range(len(mod.layers)):
    print(f"weights in {i} layer: \n {mod.layers[i].weights} \n")


mod.fit()


for i in range(len(mod.layers)):
    print(f"weights in {i} layer: \n {mod.layers[i].weights} \n")


