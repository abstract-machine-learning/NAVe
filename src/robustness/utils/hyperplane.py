import numpy as np

class Hyperplane:
    def __init__(self, coefficients=None, constant=0.0, dimensions=0):
        if coefficients is None:
            coefficients = np.zeros(dimensions)
        self.coefficients = coefficients
        self.constant = constant
    
    def __neg__(self):
        return Hyperplane(-self.coefficients, -self.constant)
    
    def __mul__(self, other):
        return Hyperplane(self.coefficients * other, self.constant * other)

    def __truediv__(self, other):
        return Hyperplane(self.coefficients / other, self.constant / other)
    
    def __call__(self, p):
        return np.dot(self.coefficients, p) + self.constant
    
    def __str__(self):
        return ' + '.join([f"{self.coefficients[i]} x_{i}" for i in range(0, self.coefficients.shape[0])]) + f" + {self.constant} = 0"