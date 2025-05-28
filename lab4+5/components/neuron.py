import math
from typing import Optional

from lab4.helpers import sigmoid, sigmoid_derivative

class Neuron:
    def __init__(self, weight: Optional[list[float]] = None, bias: Optional[float] = None):
        self.weight = weight if weight is not None else [0.0]
        self.bias = bias if bias is not None else 0.0
        self.output = 0.0
        self.inputs = []
        self.delta = 0.0

    def activate(self, inputs: list[float]) -> float:
        self.inputs = inputs
        z = sum([w * i for w, i in zip(self.weight, inputs)]) + self.bias
        self.output = sigmoid(z)
        return self.output

    def compute_hidden_delta(self, downstream_weights: list[float], downstream_deltas: list[float]) -> None:
        weighted_sum = sum(w * d for w, d in zip(downstream_weights, downstream_deltas))
        self.delta = weighted_sum * sigmoid_derivative(self.output)

    def compute_output_delta(self, target: float) -> None:
        self.delta = (target - self.output) * sigmoid_derivative(self.output)

    def update_weight(self, learning_rate: float) -> None:
        for i in range(len(self.weight)):
            self.weight[i] += learning_rate * self.delta * self.inputs[i]
        self.bias += learning_rate * self.delta
