import matplotlib.pyplot as plt
from base import *
from perceptron import *


X_train = np.random.randn(100, 3)
y_train = np.random.randn(100)

X_test = np.random.randn(20, 3)

neuron = Neuron(X_train, y_train)
neuron.train()

y_pred = neuron.predict(X_test)

print("Predictions:", y_pred)
