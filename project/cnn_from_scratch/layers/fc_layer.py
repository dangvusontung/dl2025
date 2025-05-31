import random

class FCLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # init weights and biases
        self.weights = [[random.uniform(-0.1, 0.1) for _ in range(output_size)] for _ in range(input_size)]
        self.biases = [random.uniform(-0.1, 0.1) for _ in range(output_size)]

        self.last_input = None

    def forward(self, input_vector):
        self.last_input = input_vector
        output = [0] * self.output_size

        for j in range(self.output_size):
            sum_val = 0
            for i in range(self.input_size):
                sum_val += input_vector[i] * self.weights[i][j]
            output[j] = sum_val + self.biases[j]

        return output

    def backward(self, d_output):
        # init grads
        d_weights = [[0 for _ in range(self.output_size)] for _ in range(self.input_size)]
        d_biases = list(d_output)
        d_input = [0] * self.input_size

        # compute grad w.r.t weights
        for i in range(self.input_size):
            for j in range(self.output_size):
                d_weights[i][j] = self.last_input[i] * d_output[j]

        # compute grad w.r.t input
        for i in range(self.input_size):
            sum_val = 0
            for j in range(self.output_size):
                sum_val += self.weights[i][j] * d_output[j]
            d_input[i] = sum_val

        return d_input, d_weights, d_biases
