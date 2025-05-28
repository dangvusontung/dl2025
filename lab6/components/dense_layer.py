import math
from typing import Optional

class DenseNeuron:
    def __init__(self, weight: Optional[list[float]] = None, bias: Optional[float] = None):
        self.weight = weight if weight is not None else [0.0]
        self.bias = bias if bias is not None else 0.0
        self.output = 0.0
        self.inputs = []

    def activate(self, inputs: list[float]) -> float:
        self.inputs = inputs
        z = sum([w * i for w, i in zip(self.weight, inputs)]) + self.bias
        self.output = 1 / (1 + math.exp(-z))  # Sigmoid activation
        return self.output

class DenseLayer:
    def __init__(self, neurons: list[DenseNeuron]):
        self.neurons = neurons

    def forward(self, inputs: list[float]) -> list[float]:
        return [neuron.activate(inputs) for neuron in self.neurons]
