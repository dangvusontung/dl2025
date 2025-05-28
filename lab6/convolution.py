def convolve2d(image, kernel, stride=1):
    """
    Convolution with flipped kernel, no padding, customizable stride.
    image: 2D list
    kernel: 2D list
    stride: int
    returns: 2D list (feature map)
    """
    h = len(image)
    w = len(image[0])
    kh = len(kernel)
    kw = len(kernel[0])

    output_h = (h - kh) // stride + 1
    output_w = (w - kw) // stride + 1

    flipped = [[kernel[kh - 1 - i][kw - 1 - j] for j in range(kw)] for i in range(kh)]

    output = [[0 for _ in range(output_w)] for _ in range(output_h)]

    for i in range(0, h - kh + 1, stride):
        for j in range(0, w - kw + 1, stride):
            total = 0
            for m in range(kh):
                for n in range(kw):
                    total += image[i + m][j + n] * flipped[m][n]
            output[i // stride][j // stride] = total

    return output

