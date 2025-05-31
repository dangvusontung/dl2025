from components.NeuronNetwork import NeuralNetwork, load_layer_config

def run_test(layer_size):
    layer_sizes = [2, 2, 1]
    nn = NeuralNetwork(layer_sizes, learning_rate=0.5)
    nn.print_structure()
    
    train_data = [
        ([0,0], [0]),
        ([0,1], [1]),
        ([1,0], [1]),
        ([1,1], [0]),
    ]
    
    print("\nTraining on XOR dataset (10000 epochs)...")
    nn.train(train_data, epochs=10000)
    
    print("\nResult:")
    for x, y in train_data:
        output = nn.forward(x)
        print(f"Input: {x}, Predicted: {output}, Label: {y}")

layer_size = load_layer_config("config.txt")

run_test(layer_size)