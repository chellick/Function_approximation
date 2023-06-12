from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import random
import pathlib

class Population:
    def __init__(self, search_input, left, right, step=1, p_length=32, b_length=32, x, y):
        self.population = []
        self.best = None
        self.search_input = search_input
        self.left = left
        self.right = right
        self.best_of_iter = []
        self.population_length = p_length
        self.bit_length = b_length
        self.x_val = x
        self.y_val = y

    def __getattributes__(self):
        return self.__dict__
    
    def create_population(self):
        for _ in range(self.population_length):
            self.population.append(self.create_individ(self.bit_length))
        return "Filled successfully"

    @staticmethod
    def create_individ(blength: int) -> list:
        individ = []
        for _ in range(blength):
            individ.append(random.randint(0, 1))
        return individ
    
    

x = np.linspace(-10, 10, 1)


popul = Population('max', -10, 10)
popul.create_population()
print(popul.__getattributes__())