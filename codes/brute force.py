import numpy as np
import random
import matplotlib.pyplot as plt
import math


def f(x, k, c):
    return x * k + c


def create_data_set(left, right, r, k ,c):
    arr = []
    for _ in range(r):
        point = random.uniform(left, right)
        arr.append([point, f(point, k, c)])

    return arr


education = create_data_set(-5, 5, 10, 2, 3)
test = create_data_set(-5, 5, 4, 2, 3)


k = np.arange(-10, 10, 0.1)
c = np.arange(-10, 10, 0.1)

def count_sum(sum, n):
    return (sum / n) ** 0.5

def get_odds(k, c, arr):
    summ = 0
    minim = math.inf
    n = len(education)
    for a in k:
        for b in c:
            for i in arr:
                summ += (i[1] - f(i[0], a, b)) ** 2
                if minim >= count_sum(summ, n):
                    best = (a, b)
                    minim = count_sum(summ , n)
                summ = 0
    return best, minim

res = get_odds(k, c, education)
E = res[1]

k1, c1 = res[0][0], res[0][1]

final = get_odds([k1], [c1], test)

x = []
y = []

for r in education:
    x.append(r[0])
    y.append(r[1])

x1 = np.arange(-5, 5, 0.1)
y1 = [f(i, k1, c1) for i in x1]

xt = []
yt = []

for t in test:
    xt.append(t[0])
    yt.append(t[1])


plt.scatter(x, y, color = 'red')
plt.scatter(xt, yt, color = 'green')
plt.plot(x1, y1, color = 'blue')
plt.show()
