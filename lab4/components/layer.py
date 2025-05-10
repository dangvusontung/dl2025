from lab4.components.neuron import Neuron
import typing

class Layer: 
    def __init__(self, neurons: list[Neuron]):
        self.neurons = neurons

    def activate(self, inputs: list[float]) -> list[float]:
        return [neuron.activate(inputs) for neuron in self.neurons]
    
    def forward(self, inputs: list[float]) -> list[float]:
        return self.activate(inputs)
    
