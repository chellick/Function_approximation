import numpy as np
import matplotlib.pyplot as plt


class Data:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.xp = []
        self.yp = []


    def add_point(self, px):
        py = self.interpolate(self.x, self.y, px)

        self.x = np.append(self.x, px)
        self.x = np.sort(self.x)
        i = np.where(self.x == px)[0]
        self.y = np.insert(self.y, i, py)

        self.xp.append(px)
        self.yp.append(py)

    def get_data(self):
        return self.x, self.y, self.xp, self.yp


    @staticmethod
    def interpolate(arr, arr2, x_new):
        index = np.searchsorted(arr, x_new, side='left')
        x_left, x_right = arr[index - 1], arr[index]
        y_left, y_right = arr2[index - 1], arr2[index]

        k = (y_right - y_left) / (x_right - x_left)
        b = y_left - k * x_left
        y_new = k * x_new + b
        return y_new


class Point:
    def __init__(self, x):
        self.x = x

    def get_init(self):
        return self.x


x = np.array([0, 1, 2, 3, 4, 5, 6])
y = np.array([-2.0, 3.89, 4.033, 5.023, 9.0, 11.0, 9])
data = Data(x, y)
point = Point(5.4)


data.add_point(point.get_init())

plt.plot(data.get_data()[0], data.get_data()[1], "bo")
plt.plot(data.get_data()[2], data.get_data()[3], "ro")
plt.show()
