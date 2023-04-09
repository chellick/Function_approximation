
class Vector(list):
    def __init__(self, array):
        super().__init__(array)

    def __add__(self, other):
        return Vector(a + b for a, b in zip(self, other))
    
    def __sub__(self, other):
        return Vector(a - b for a, b in zip(self, other))

    def __mul__(self, other):
        return Vector(a * b for a, b in zip(self, other))
    
    def __truediv__(self, other):
        return Vector(a / b for a, b in zip(self, other))
    
    def __eq__(self, other):
        return all(a == b for a, b in zip(self, other))
    
    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return '[' + ', '.join(map(str, self)) + ']'
    
    def outer(self, other):
        result = Matrix.zeros(len(self), len(other))
        for i in range(len(result.data)):
            for j in range(len(result.data)):
                result.data[i][j] = self[i] * other[j]
        return result



class Matrix:
    def __init__(self, data):
        self.data = [Vector(row) for row in data]

    def __add__(self, other):
        return Matrix([row1 + row2 for row1, row2 in zip(self.data, other.data)])

    def __sub__(self, other):
        return Matrix([row1 - row2 for row1, row2 in zip(self.data, other.data)])

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if len(self.data[0]) != len(other.data):
                print(len(self.data[0]), len(other.data))
                raise ValueError("Matrix dimensions do not match for multiplication.")
            result = Matrix.zeros(len(self.data), len(other.data[0]))
            for i in range(len(self.data)):
                for j in range(len(other.data[0])):
                    for k in range(len(other.data)):
                        result.data[i][j] += self.data[i][k] * other.data[k][j]
            return result
        else:
            return Matrix([row * other for row in self.data])

    def __truediv__(self, other):
        return Matrix([row / other for row in self.data])

    def __eq__(self, other):
        return all(row1 == row2 for row1, row2 in zip(self.data, other.data))

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return '\n'.join(map(str, self.data))
    
    def shape(self):
        rows = len(self.data)
        cols = len(self.data[0]) if rows > 0 else 0
        return (rows, cols)

    def T(self):
        result = Matrix.zeros(self.shape()[0], self.shape()[1])
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                result.data[j][i] = self.data[i][j]
        return result   

    @staticmethod
    def zeros(rows, cols):
        return Matrix([[0] * cols for _ in range(rows)])








