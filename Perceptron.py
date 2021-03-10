# Task 2 - Implement a binary perceptron
# Task 3 - Use the binary perceptron to train classifiers:
#     * class 1 and class 2
#     * class 2 and class 3, and 
#     * class 1 and class 3

import numpy as np
class Perceptron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def perceptron(self, train_data, epoch):
        weight = [self.weight for i in range(len(train_data))]
        bias = self.bias
        for i in range(epoch):
            for row in dataset:
                activation = np.dot(row, weight) + bias
                if y * activation <= 0:




if __name__ == "__main__":
    perceptron = Perceptron(0, 1)
    