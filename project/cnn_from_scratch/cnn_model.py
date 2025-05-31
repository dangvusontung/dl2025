# cnn_model.py

from layers.conv_layer import ConvLayer
from layers.pool_layer import PoolLayer
from layers.fc_layer import FCLayer
from layers.activations import ReLULayer, softmax_forward
from config_reader import read_config, calculate_fc_input_size

class CNN:
    def __init__(self, config_path="config.txt"):
        # Reading config – hoping it won't crash
        self.config = read_config(config_path)

        # Compute FC input size – please work
        fc_input_size = calculate_fc_input_size(self.config)

        # Conv block 1 – trust the config
        self.conv1 = ConvLayer(
            num_filters=self.config['conv1_filters'],
            kernel_size=self.config['conv1_kernel_size'],
            input_channels=self.config['conv1_input_channels']
        )
        self.relu1 = ReLULayer()
        self.pool1 = PoolLayer(
            kernel_size=self.config['pool_kernel_size'],
            stride=self.config['pool_stride']
        )

        # Conv block 2 – no turning back
        self.conv2 = ConvLayer(
            num_filters=self.config['conv2_filters'],
            kernel_size=self.config['conv2_kernel_size'],
            input_channels=self.config['conv1_filters']  # seems correct?
        )
        self.relu2 = ReLULayer()
        self.pool2 = PoolLayer(
            kernel_size=self.config['pool_kernel_size'],
            stride=self.config['pool_stride']
        )

        # Final FC layer – that's all folks
        self.fc1 = FCLayer(
            input_size=fc_input_size,
            output_size=self.config['fc_output_size']
        )

        # Layers with trainable params
        self.layers = [self.conv1, self.conv2, self.fc1]

        # Stash for unflatten
        self.last_conv_output_shape = None

    def forward(self, input_image):
        """
        input_image: list of shape 1x28x28
        """
        x = self.conv1.forward(input_image)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        # Flatten – brute force way
        self.last_conv_output_shape = (len(x), len(x[0]), len(x[0][0]))
        flat = []
        for c in x:
            for row in c:
                for val in row:
                    flat.append(val)

        logits = self.fc1.forward(flat)
        probs = softmax_forward(logits)

        return probs

    def backward(self, d_output):
        """
        d_output: gradient from loss (len 10)
        """
        # FC backward
        d_fc1_input, d_fc1_W, d_fc1_b = self.fc1.backward(d_output)

        # Unflatten gradient back to conv2 output shape
        d_flat = d_fc1_input
        channels, height, width = self.last_conv_output_shape

        d_conv2_out = []
        idx = 0
        for c in range(channels):
            channel = []
            for r in range(height):
                row = []
                for col in range(width):
                    row.append(d_flat[idx])
                    idx += 1
                channel.append(row)
            d_conv2_out.append(channel)

        d_pool2 = self.pool2.backward(d_conv2_out)
        d_relu2 = self.relu2.backward(d_pool2)
        d_conv2_input, d_conv2_filters, d_conv2_biases = self.conv2.backward(d_relu2)

        d_pool1 = self.pool1.backward(d_conv2_input)
        d_relu1 = self.relu1.backward(d_pool1)
        d_conv1_input, d_conv1_filters, d_conv1_biases = self.conv1.backward(d_relu1)

        grads = [
            (d_conv1_filters, d_conv1_biases),
            (d_conv2_filters, d_conv2_biases),
            (d_fc1_W, d_fc1_b),
        ]
        return grads
