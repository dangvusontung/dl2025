import typing

from lab4.components.neuron import Neuron


class NeuronNetwork: 
    def __init__(self, layers: list[typing.List[Neuron]]):
        self.layers = layers

    def forward(self, inputs: list[float]) -> list[float]:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs