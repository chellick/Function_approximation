import numpy as np
from perceptron import *
# Задаем параметры функции
k = 2
b = 3

# Генерируем значения x и вычисляем соответствующие значения функции
x = np.linspace(-1, 1, 100)
y = k * (x ** 2) + b

# Формируем датасет
dataset = np.column_stack((x, np.full_like(x, k), np.full_like(x, b), y))

# Перемешиваем датасет
np.random.shuffle(dataset)

# Разделяем на обучающую и тестовую выборки
train_data = dataset[:80, :3]
train_labels = dataset[:80, 3]
test_data = dataset[80:, :3]
test_labels = dataset[80:, 3]


n = Neuron(train_data, train_labels)

print(n.weights1, "\n","\n",  n.weights2, "\n", "\n", n.weights3)
