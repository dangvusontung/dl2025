import typing
from lab4.components.neuron import Neuron
from lab4.components.layer import Layer

class NeuronNetwork: 
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, inputs: list[float]) -> list[float]:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, targets: list[float]) -> None:
        self.layers[-1].compute_output_deltas(targets)

        for i in range(len(self.layers) - 2, -1, -1):
            downstream_weights = self.layers[i + 1].get_weights()
            downstream_deltas = [neuron.delta for neuron in self.layers[i + 1].neurons]
            self.layers[i].backward(downstream_weights, downstream_deltas)

    def update_weights(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.update_weights(learning_rate)

    def train(self, inputs: list[float], targets: list[float], learning_rate: float) -> None:
        self.forward(inputs)
        self.backward(targets)
        self.update_weights(learning_rate)