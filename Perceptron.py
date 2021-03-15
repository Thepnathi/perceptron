# Task 2 - Implement a binary perceptron
# Task 3 - Use the binary perceptron to train classifiers:

import numpy as np
from DatasetLoader import DatasetLoader

class Perceptron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def train_perceptron(self, train_data, label_data, num_epoch=20):
        weight = np.zeros(len(train_data[0]))    
        bias = 0                                                  
        for _ in range(num_epoch):
            for i, row in enumerate(train_data):
                activation = np.dot(weight, row) + bias
                if label_data[i] * np.sign(activation) <= 0:  
                    weight = self.compute_new_weight(weight, row, label_data[i])          # Update weights and bias 
                    bias = bias + label_data[i]
        return bias, weight

    def compute_new_weight(self, oldWeight, row, class_value):
        newRow = np.multiply(class_value, row)
        return np.add(oldWeight, newRow)

if __name__ == "__main__":
    # Loads the randomised dataset with two classes or all
    dataloader = DatasetLoader()
    train_data, train_label = dataloader.extract_two_classes_from_dataset('train.data', 1, 2)
    test_data, test_label = dataloader.extract_two_classes_from_dataset('test.data', 1, 2)
    perceptron = Perceptron(0, 0)

    bias, weight = perceptron.train_perceptron(train_data, train_label, 20)
    print(bias)
    print(weight)
