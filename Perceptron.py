# Task 2 - Implement a binary perceptron
# Task 3 - Use the binary perceptron to train classifiers:

import numpy as np
from DatasetHandler import DatasetHandler
from typing import Tuple

class Perceptron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    # Train the perceptron model given feature and corresponding label dataset. Model can be trained for n number of iterations
    def train_perceptron(self, train_data, label_data, num_epoch=20) -> Tuple[int, list[float]]:
        weight = np.zeros(len(train_data[0]))    
        bias = 0                                                  
        for _ in range(num_epoch):
            for i, row in enumerate(train_data):
                activation = np.dot(weight, row) + bias
                if label_data[i] * np.sign(activation) <= 0:  
                    weight = self.compute_new_weight(weight, row, label_data[i]) 
                    bias = bias + label_data[i]
        self.weight, self.bias = weight, bias

    def compute_new_weight(self, oldWeight, row, class_value):
        newRow = np.multiply(class_value, row)
        return np.add(oldWeight, newRow)

    def get_weight(self):
        return self.weight

    def get_bias(self):
        return self.bias

if __name__ == "__main__":
    # Loads the test dataset for class 1 and class 2 in order 
    test1 = DatasetHandler()
    test1.extract_two_classes_from_dataset('test.data', 1, 2)

    # Initialise the Perceptron object with 0 weight and 0 bias
    # Train a model given test1 dataset
    model1 = Perceptron(0, 0)
    model1.train_perceptron(test1.feature_dataset, test1.label_dataset, 20)

    print("Perceptron model with test dataset for class 1 and 2...")
    print(f"Bias is {model1.get_bias()} and Weight of {model1.get_weight()}")