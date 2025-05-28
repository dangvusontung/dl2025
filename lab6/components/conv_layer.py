from convolution import convolve2d

class Conv2D:
    def __init__(self, kernel, stride=1):
        """
        kernel: 2D list (filter)
        stride: int
        """
        self.kernel = kernel
        self.stride = stride

    def forward(self, image):
        """
        image: 2D list
        return: 2D list (feature map)
        """
        return convolve2d(image, self.kernel, self.stride)