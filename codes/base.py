


class Vector:
    def __init__(self, array):
        self.array = array

    def __getattributes__(self):
        return self.__dict__
    
    def __add__(self, other):
        result = Vector([])
        result.array = list(a + b for a, b in zip(self.array, other.array))
        return result.array
    
    def __sub__(self, other):
        result = Vector([])
        result.array = list(a - b for a, b in zip(self.array, other.array))
        return result.array

    def __mul__(self, other):
        result = Vector([])
        result.array = list(a * b for a, b in zip(self.array, other.array))
        return result.array
    
    def __truediv__(self, other):
        result = Vector([])
        result.array = list(a / b for a, b in zip(self.array, other.array))
        return result.array
    
    def __eq__(self, other):
        for a, b in zip(self.array, other.array):
            if a == b:
                result = True
            else: 
                result = False
                return result
        return result
    
    def __ne__(self, other):
        for a, b in zip(self.array, other.array):
            if a == b:
                result = False
            else: 
                result = True
                return result
        return result


a = Vector([5, 6, 7])
b = Vector([5, 6, 7])
print(a == b)
print(a != b)



# class Neuron:
#     def __init__(self, inputs, weights, errors) -> None:
#         self.inputs = inputs
#         self.weights = weights
#         self.errors = errors

#     def __getattributes__(self):
#         return self.__dict__
    
#     # def 


