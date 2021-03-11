# Task 2 - Implement a binary perceptron
# Task 3 - Use the binary perceptron to train classifiers:
#     * class 1 and class 2
#     * class 2 and class 3, and 
#     * class 1 and class 3

import numpy as np
from DatasetLoader import DatasetLoader
class Perceptron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def train_perceptron(self, train_data, num_epoch):
        weight = [self.weight for i in range(len(train_data[0]))]
        bias = self.bias
        for i in range(num_epoch):
            for row in train_data:
                activation = self.compute_activation(row, weight, bias)
                label = row[-1]
                if label * activation <= 0:           
                    weight = self.compute_new_weight(weight, row, label)          # Update weights and bias
                    bias += label
        return bias, weight

    def compute_activation(self, dataRow, weight, bias):
        return np.inner(dataRow, weight) + bias

    def compute_new_weight(self, oldWeight, row, class_value):
        newRow = np.multiply(class_value, row)
        return np.add(oldWeight, newRow)

if __name__ == "__main__":
    # Loads the randomised dataset with two classes or all
    dataloader = DatasetLoader()
    train_data = dataloader.extract_two_classes_from_dataset('train.data', 1, 2)
    perceptron = Perceptron(0, 0)

    bias, weight = perceptron.train_perceptron(train_data, 5)
    print(bias)
    print(weight)


    