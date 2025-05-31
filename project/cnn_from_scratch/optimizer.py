# optimizer.py
from layers.conv_layer import ConvLayer
from layers.fc_layer import FCLayer

class GradientDescent:
    def __init__(self, model_layers, learning_rate=0.03):
        self.layers = [layer for layer in model_layers if isinstance(layer, (ConvLayer, FCLayer))]
        self.learning_rate = learning_rate

    def step(self, param_gradients):
        if len(param_gradients) != len(self.layers):
            raise ValueError("Số lượng gradients không khớp với số lớp có tham số.")

        for i, layer in enumerate(self.layers):
            dW, db = param_gradients[i]

            # Cập nhật bias
            for j in range(len(layer.biases)):
                layer.biases[j] -= self.learning_rate * db[j]

            # Cập nhật weights hoặc filters
            if isinstance(layer, FCLayer):
                for r in range(len(layer.weights)):
                    for c in range(len(layer.weights[0])):
                        layer.weights[r][c] -= self.learning_rate * dW[r][c]
            else:  # ConvLayer
                for f in range(len(layer.filters)):
                    for c in range(len(layer.filters[0])):
                        for r in range(len(layer.filters[0][0])):
                            for col in range(len(layer.filters[0][0][0])):
                                layer.filters[f][c][r][col] -= self.learning_rate * dW[f][c][r][col]
