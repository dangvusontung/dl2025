import random
from lab4.components.neuron import Neuron
from lab4.components.layer import Layer
from lab4.components.NeuronNetwork import NeuronNetwork

def create_network():
    # Create a network with 2 input neurons, 2 hidden neurons, and 1 output neuron
    # Input layer (2 neurons)
    input_layer = Layer([
        Neuron([random.uniform(-1, 1), random.uniform(-1, 1)], random.uniform(-1, 1))
        for _ in range(2)
    ])
    
    # Hidden layer (2 neurons)
    hidden_layer = Layer([
        Neuron([random.uniform(-1, 1), random.uniform(-1, 1)], random.uniform(-1, 1))
        for _ in range(2)
    ])
    
    # Output layer (1 neuron)
    output_layer = Layer([
        Neuron([random.uniform(-1, 1), random.uniform(-1, 1)], random.uniform(-1, 1))
    ])
    
    return NeuronNetwork([input_layer, hidden_layer, output_layer])

def train_xor(network, epochs=10000, learning_rate=0.1):
    # XOR training data
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]
    
    # Training loop
    for epoch in range(epochs):
        total_error = 0
        for inputs, targets in training_data:
            # Forward pass
            outputs = network.forward(inputs)
            
            # Calculate error
            error = sum((t - o) ** 2 for t, o in zip(targets, outputs)) / len(targets)
            total_error += error
            
            # Backward pass and weight update
            network.train(inputs, targets, learning_rate)
        
        # Print progress every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}, Average Error: {total_error/len(training_data):.6f}")

def test_network(network):
    # Test cases
    test_cases = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0)
    ]
    
    print("\nTesting the trained network:")
    print("Input\t\tExpected\tOutput\t\tError")
    print("-" * 50)
    
    for inputs, expected in test_cases:
        output = network.forward(inputs)[0]
        error = abs(expected - output)
        print(f"{inputs}\t\t{expected}\t\t{output:.4f}\t\t{error:.4f}")

def main():
    print("Creating and training a neural network for XOR problem...")
    network = create_network()
    
    print("\nTraining the network...")
    train_xor(network)
    
    # Test the trained network
    test_network(network)

if __name__ == "__main__":
    main() 