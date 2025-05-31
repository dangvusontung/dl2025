class PoolLayer:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.last_input = None

    def forward(self, input_map):
        self.last_input = input_map
        channels = len(input_map)
        in_h = len(input_map[0])
        in_w = len(input_map[0][0])
        out_h = (in_h - self.kernel_size) // self.stride + 1
        out_w = (in_w - self.kernel_size) // self.stride + 1

        output = [[[0 for _ in range(out_w)] for _ in range(out_h)] for _ in range(channels)]
        for c in range(channels):
            for y in range(out_h):
                for x in range(out_w):
                    start_y, start_x = y * self.stride, x * self.stride
                    window = [row[start_x : start_x + self.kernel_size] for row in input_map[c][start_y : start_y + self.kernel_size]]
                    max_val = -float('inf')
                    for row in window:
                        for val in row:
                            if val > max_val:
                                max_val = val
                    output[c][y][x] = max_val
        return output

    def backward(self, grad_output):
        channels = len(self.last_input)
        in_h = len(self.last_input[0])
        in_w = len(self.last_input[0][0])
        out_height = len(grad_output[0])
        out_width = len(grad_output[0][0])
        d_input = [[[0 for _ in range(in_w)] for _ in range(in_h)] for _ in range(channels)]
        for c in range(channels):
            for y in range(out_height):
                for x in range(out_width):
                    start_y, start_x = y * self.stride, x * self.stride
                    max_val = -float('inf')
                    max_y_local, max_x_local = -1, -1
                    for ky in range(self.kernel_size):
                        for kx in range(self.kernel_size):
                            current_val = self.last_input[c][start_y + ky][start_x + kx]
                            if current_val > max_val:
                                max_val = current_val
                                max_y_local, max_x_local = start_y + ky, start_x + kx
                    if max_y_local != -1:
                        d_input[c][max_y_local][max_x_local] += grad_output[c][y][x]
        return d_input
