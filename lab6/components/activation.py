import math

class ReLU:
    def forward(self, inputs):
        """
        inputs: list of floats (1D or 2D)
        return: same shape with ReLU applied
        """
        if isinstance(inputs[0], list):  # 2D case
            return [[max(0, val) for val in row] for row in inputs]
        else:  # 1D case
            return [max(0, val) for val in inputs]

class Softmax:
    def forward(self, inputs):
        """
        inputs: 1D list
        return: 1D list of softmax probabilities
        """
        max_input = max(inputs)  # for numerical stability
        exps = [math.exp(i - max_input) for i in inputs]
        sum_exps = sum(exps)
        return [j / sum_exps for j in exps]
