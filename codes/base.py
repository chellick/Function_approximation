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

    # def __repr__(self):
    #     return '[' + ', '.join(map(str, self)) + ']'

# Пример использования
v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])

# print(v1)  # Вывод: [1, 2, 3]
# print(v1 + v2)  # Вывод: [5, 7, 9]

    
    

class Matrix:
    def __init__(self, array):
        self.array = array

    def __getattributes__(self):
        return self.__dict__


a = Vector([8, 6, 7])
b = Vector([5, 6, 7])
c = Matrix([a, b])


print(len(a))
print(c.array)




# class Neuron:
#     def __init__(self, inputs, weights, errors) -> None:
#         self.inputs = inputs
#         self.weights = weights
#         self.errors = errors

#     def __getattributes__(self):
#         return self.__dict__
    
#     # def 


