from components.conv_layer import Conv2D
from components.maxpool_layer import MaxPool2D
from components.flatten_layer import Flatten
from components.dense_layer import DenseLayer, DenseNeuron
from components.activation import ReLU, Softmax

class SimpleCNN:
    def __init__(self):
        # NOTE: kernel and weights are hardcoded for demo
        kernel = [[1, 0, -1],
                  [1, 0, -1],
                  [1, 0, -1]]
        self.conv = Conv2D(kernel=kernel, stride=1)
        self.relu = ReLU()
        self.pool = MaxPool2D(size=2, stride=2)
        self.flatten = Flatten()

        dummy_input_size = 144  # ví dụ từ 12x12 ảnh sau conv/pool
        self.dense = DenseLayer([
            DenseNeuron(weight=[0.01] * dummy_input_size, bias=0.0)
            for _ in range(4)
        ])
        self.softmax = Softmax()

    def forward(self, image):
        x = self.conv.forward(image)
        x = self.relu.forward(x)
        x = self.pool.forward(x)
        x = self.flatten.forward(x)
        x = self.dense.forward(x)
        x = self.softmax.forward(x)
        return x
