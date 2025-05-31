import random

class ConvLayer:
    def __init__(self, num_filters, kernel_size, input_channels):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_channels = input_channels

        self.filters = [
            [
                [
                    [random.uniform(-0.1, 0.1) for _ in range(kernel_size)] 
                    for _ in range(kernel_size)
                ] for _ in range(input_channels)
            ] for _ in range(num_filters)
        ]
        
        self.biases = [random.uniform(-0.1, 0.1) for _ in range(num_filters)]
        
        self.last_input = None
        
    def forward(self, input_data):
        self.last_input = input_data
        
        channels, in_h, in_w = len(input_data), len(input_data[0]), len(input_data[0][0])
        
        out_h = in_h - self.kernel_size + 1
        out_w = in_w - self.kernel_size + 1
        
        output = [[[0 for _ in range(out_w)] for _ in range(out_h)] for _ in range(self.num_filters)]

        for f in range(self.num_filters):
            for y in range(out_h):
                for x in range(out_w):
                    receptive_field_sum = 0
                    for c in range(channels):
                        for ky in range(self.kernel_size):
                            for kx in range(self.kernel_size):
                                receptive_field_sum += input_data[c][y + ky][x + kx] * self.filters[f][c][ky][kx]
                    
                    output[f][y][x] = receptive_field_sum + self.biases[f]
        
        return output

    def backward(self, d_output):
        d_filters = [[[([0] * self.kernel_size) for _ in range(self.kernel_size)] for _ in range(self.input_channels)] for _ in range(self.num_filters)]
        d_biases = [0] * self.num_filters
        d_input = [[[0] * len(self.last_input[0][0]) for _ in range(len(self.last_input[0]))] for _ in range(self.input_channels)]

        channels, in_h, in_w = len(self.last_input), len(self.last_input[0]), len(self.last_input[0][0])
        out_h, out_w = len(d_output[0]), len(d_output[0][0])

        for f in range(self.num_filters):
            for y in range(out_h):
                for x in range(out_w):
                    d_biases[f] += d_output[f][y][x]
                    
                    for c in range(channels):
                        for ky in range(self.kernel_size):
                            for kx in range(self.kernel_size):
                                d_filters[f][c][ky][kx] += self.last_input[c][y + ky][x + kx] * d_output[f][y][x]
                                d_input[c][y + ky][x + kx] += self.filters[f][c][ky][kx] * d_output[f][y][x]
        
        return d_input, d_filters, d_biases