import numpy as np
from perceptron import *
# Задаем параметры функции
k = 2
b = 3

# Генерация данных
x = np.linspace(-1, 1, 100)
y = k * (x ** 2) + b

# Разбиение данных на обучающую и тестовую выборки
train_size = int(0.8 * len(x))

x_train = x[:train_size].reshape(-1, 1)
y_train = y[:train_size].reshape(-1, 1)
x_test = x[train_size:].reshape(-1, 1)
y_test = y[train_size:].reshape(-1, 1)


 
n = Neuron(x_train, y_train)
print(n.forward(n.x_true))
n.train()
