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
    def perceptron(self, train_data, num_epoch):
        weight = [self.weight for i in range(len(train_data[0]))]
        bias = self.bias
        for i in range(num_epoch):
            for row in train_data:
                activation_score = np.dot(row[:-1], weight) + bias
                label = row[-1]
                if label * activation_score <= 0:           # if score <= 0, we made miscalculation
                    weight = weight + label * row           # Update weights and bias
                    bias += label
        return bias, weight

    def predict(self, test_data):
        pass
    # activation_score = w1x1 + w2x2, w3x3...wdxd


if __name__ == "__main__":
    perceptron = Perceptron(0, 0)
    