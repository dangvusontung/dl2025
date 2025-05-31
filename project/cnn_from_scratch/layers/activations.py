import math

def relu_forward(input_data):
    if not input_data: return []
    if isinstance(input_data[0], (int, float)):
        return [max(0, x) for x in input_data]
    else:
        output = []
        for channel in input_data:
            new_channel = []
            for row in channel:
                new_channel.append([max(0, x) for x in row])
            output.append(new_channel)
        return output

def relu_backward(d_output, original_input):
    if not d_output: return []
    if isinstance(d_output[0], (int, float)):
        return [d * (1 if o > 0 else 0) for d, o in zip(d_output, original_input)]
    else:
        d_input = []
        for c in range(len(d_output)):
            d_channel = []
            for i in range(len(d_output[c])):
                row = []
                for j in range(len(d_output[c][i])):
                    grad = d_output[c][i][j] * (1 if original_input[c][i][j] > 0 else 0)
                    row.append(grad)
                d_channel.append(row)
            d_input.append(d_channel)
        return d_input

def softmax_forward(vector):
    max_val = max(vector)
    exps = [math.exp(x - max_val) for x in vector]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

class ReLULayer:
    def __init__(self):
        self.last_input = None

    def forward(self, input_data):
        self.last_input = input_data
        return relu_forward(input_data)

    def backward(self, d_output):
        return relu_backward(d_output, self.last_input)
