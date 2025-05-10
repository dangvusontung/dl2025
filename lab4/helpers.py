import math

def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))

def sigmoid_derivative(z: float) -> float:
    return z * (1 - z)