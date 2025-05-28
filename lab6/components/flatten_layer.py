class Flatten:
    def forward(self, input_2d):
        """
        input_2d: 2D list
        return: 1D list (flattened)
        """
        return [val for row in input_2d for val in row]
