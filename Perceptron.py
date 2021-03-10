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

    # Train the Perceptron model
    def perceptron(self, train_data, epoch):
        weight = [self.weight for i in range(len(train_data))]
        bias = self.bias
        for i in range(epoch):
            for row in train_data:
                activation_score = np.dot(row, weight) + bias
                label = row[-1]
                if label * activation_score <= 0:
                    weight = weight + label * row                     # Update weights and bias
                    bias += label
        return bias, weight


if __name__ == "__main__":
    perceptron = Perceptron(0, 0)
    