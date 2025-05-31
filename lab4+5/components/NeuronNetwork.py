from helpers import sigmoid, sigmoid_derivative

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        for i in range(1, self.num_layers):
            w = []
            for j in range(layer_sizes[i]):
                w.append([1.0 for k in range(layer_sizes[i-1])])
            self.weights.append(w)
        
        for i in range(1, self.num_layers):
            self.biases.append([1.0 for j in range(layer_sizes[i])])
    
    def forward(self, x):
        self.activations = [x[:]]
        self.z_values = []
        
        a = x[:]
        for l in range(self.num_layers - 1):
            z = []
            a_next = []
            for i in range(self.layer_sizes[l+1]):
                zi = sum(self.weights[l][i][j] * a[j] for j in range(self.layer_sizes[l])) + self.biases[l][i]
                z.append(zi)
                a_next.append(sigmoid(zi))
            self.z_values.append(z)
            self.activations.append(a_next)
            a = a_next
        return a
    
    def backward(self, x, y):
        self.forward(x)
        
        delta = []
        for l in range(self.num_layers - 1):
            delta.append([0.0] * self.layer_sizes[l + 1])
        
        output_layer = self.num_layers - 2
        for i in range(self.layer_sizes[-1]):
            error = self.activations[-1][i] - y[i]
            delta[output_layer][i] = error * sigmoid_derivative(self.z_values[output_layer][i])
        
        for l in range(output_layer - 1, -1, -1):
            for i in range(self.layer_sizes[l + 1]):
                error = 0.0
                for j in range(self.layer_sizes[l + 2]):
                    error += delta[l + 1][j] * self.weights[l + 1][j][i]
                delta[l][i] = error * sigmoid_derivative(self.z_values[l][i])
        
        for l in range(self.num_layers - 1):
            for i in range(self.layer_sizes[l + 1]):
                for j in range(self.layer_sizes[l]):
                    self.weights[l][i][j] -= self.learning_rate * delta[l][i] * self.activations[l][j]
                self.biases[l][i] -= self.learning_rate * delta[l][i]
    
    def train(self, training_data, epochs):
        for epoch in range(epochs):
            for x, y in training_data:
                self.backward(x, y)
    
    def print_structure(self):
        for l in range(self.num_layers-1):
            print(f"Layer {l+1} -> {l+2}:")
            for row in self.weights[l]:
                print(row)
            print(self.biases[l])

def load_layer_config(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    n_layers = int(lines[0])
    layer_sizes = [int(x.strip()) for x in lines[1:n_layers+1]]
    return layer_sizes