import numpy as np
import matplotlib.pyplot as plt


class Spline:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.xp = []
        self.yp = []

    def get_data(self):
        return self.x, self.y, self.xp, self.yp
    
    def spline(self):
        d = []
        n = len(self.x)
        h = [self.x[i + 1] - self.x[i] for i in range(n - 1)]
        b = [(self.y[i + 1] - self.y[i]) / h[i] for i in range(n - 1)]
        alpha = [0] * n
        
        for i in range(1, n - 1):
            alpha[i] = 3 * (b[i] - b[i - 1])

        c = [0] * n
        l = [0] * n
        mu = [0] * n
        z = [0] * n

        l[0] = 1
        mu[0] = 0
        z[0] = 0

        l[n - 1] = 1
        z[n - 1] = 0
        c[n - 1] = 0

        for el in range(n-2, -1, -1):
            c[el] = z[el] - mu[el] * c[el + 1]
            b[el] = (self.y[el + 1] - self.y[el]) / h[el] - h[el] * (c[el + 1] + 2 * c[el]) / 3
            d.append((c[el + 1] - c[el]) / (3 * h[el]))

        a = self.y[:-1]
        res = []

        for i in range(len(self.x) - 1):
            xs = np.linspace(self.x[i], self.x[i+1], 1000)
            ys = a[i] + b[i]*(xs - self.x[i]) + c[i]*(xs - self.x[i])**2 + d[i]*(xs - self.x[i])**3
            plt.plot(xs, ys)
    
def f(x):
    return 0.01 * np.cos(x ** 0.5 + np.sin(x))
    # return x ** 2

x = np.linspace(-10, 10, 50)

data = Spline(x, f(x))
plt.plot(x, f(x), "bo")
data.spline()
data.get_data()
# plt.title('spline')
# plt.legend()
plt.show()                             #TODO: написать README

