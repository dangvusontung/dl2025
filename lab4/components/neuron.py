import math
from typing import Optional

from lab4.helpers import sigmoid


class Neuron:
    def __init__(self, weight: Optional[float] = None, bias: Optional[float] = None):
        self.weight = weight
        self.bias = bias

    def activate(self, inputs: list[float]) -> float:
        z = sum([w * i for w, i in zip(self.weight, inputs)]) + self.bias
        return sigmoid(z)
    