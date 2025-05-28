from lab4.components.neuron import Neuron
import typing

class Layer: 
    def __init__(self, neurons: list[Neuron]):
        self.neurons = neurons

    def activate(self, inputs: list[float]) -> list[float]:
        return [neuron.activate(inputs) for neuron in self.neurons]
    
    def forward(self, inputs: list[float]) -> list[float]:
        return self.activate(inputs)
    
    def get_output(self) -> list[float]:
        return [neuron.output for neuron in self.neurons]
    
    def get_weights(self) -> list[list[float]]:
        return [neuron.weight for neuron in self.neurons]
    
    def get_biases(self) -> list[float]: 
        return [neuron.bias for neuron in self.neurons]
    
    def update_weights(self, learning_rate: float) -> None:
        for neuron in self.neurons:
            neuron.update_weight(learning_rate)

    def backward(self, downstream_weights: list[list[float]], downstream_deltas: list[float]) -> None:
        for i, neuron in enumerate(self.neurons):
            neuron.compute_hidden_delta(
                [w[i] for w in downstream_weights], downstream_deltas
            )

    def compute_output_deltas(self, targets: list[float]) -> None:
        for neuron, target in zip(self.neurons, targets):
            neuron.compute_output_delta(target)
