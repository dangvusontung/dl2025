import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)