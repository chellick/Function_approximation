import numpy as np
import matplotlib.pyplot as plt

class Curve:
    def __init__(self, points, num=1000):
        self.points = points
        self.num = num

    @staticmethod
    def binomial(n, k):
        return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k)) # type: ignore
    
    def bezier_curve(self):
        t = np.linspace(0, 1, num=self.num)
        n = len(points) - 1
        curve = np.zeros((self.num, 2))
        for i in range(self.num):
            for j in range(n+1):
                curve[i] += self.binomial(n, j) * ((1 - t[i]) ** (n-j)) * (t[i] ** j) * points[j]
        return curve
    

x = np.array([0, 1, 2, 3, 4, 5, 6])
y = np.array([-2.0, 3.89, 4.033, 5.023, 9.0, 11.0, 9])
points = []

for i in range(len(x)):
    points.append((x[i], y[i]))

points = np.array(points)



c = Curve(points=points)
curve = c.bezier_curve()

plt.plot(points[:, 0], points[:, 1], 'o', label="Узлы интерполяции")
plt.plot(curve[:, 0], curve[:, 1], label="Кривая Безье")
plt.legend()
plt.show()
