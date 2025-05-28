class MaxPool2D:
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def forward(self, image):
        """
        image: 2D list
        return: 2D list after max pooling
        """
        h = len(image)
        w = len(image[0])
        out_h = (h - self.size) // self.stride + 1
        out_w = (w - self.size) // self.stride + 1

        output = [[0 for _ in range(out_w)] for _ in range(out_h)]

        for i in range(0, h - self.size + 1, self.stride):
            for j in range(0, w - self.size + 1, self.stride):
                max_val = float('-inf')
                for m in range(self.size):
                    for n in range(self.size):
                        val = image[i + m][j + n]
                        if val > max_val:
                            max_val = val
                output[i // self.stride][j // self.stride] = max_val

        return output
