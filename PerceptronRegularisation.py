# Task 2 - Implement a binary perceptron
# Task 3 - Use the binary perceptron to train classifiers:

import numpy as np
from DatasetHandler import DatasetHandler
from Perceptron import Perceptron

class PerceptronRegularisation(Perceptron):
    def __init__(self, weight, bias):
        super(weight, bias)

    # Train the perceptron model given feature and corresponding label dataset. Model can be trained for n number of iterations
    # In our case the learning rate will be at 1 for Perceptron
    def train_perceptron_l2(self, feature_dataset, label_dataset, coefficient, num_epoch=20):
        weight = np.zeros(len(feature_dataset[0]))    
        bias = 0                                                  
        for _ in range(num_epoch):
            for i, row in enumerate(feature_dataset):
                activation = np.dot(weight, row) + bias
                if label_dataset[i] * np.sign(activation) <= 0:  
                    weight = self.apply_l2_regularisation(weight, row, label_dataset[i], coefficient)
                    bias = bias + label_dataset[i]
        self.weight, self.bias = weight, bias

    # Returns the new weight after we apply the stochastic gradient update and regularisation term
    # W <- W - learning_rate(-yi * Xi + 2*coefficient*W)
    def apply_l2_regularisation(self, old_weight, row, class_value, coefficient):
        new_weight = np.multiply((1-2*coefficient), old_weight)             # W = (1-2cofficient) * W + yi * Xi
        result = np.add(new_weight, np.multiply(class_value, row))
        return result


if __name__ == "__main__":
    # Loads the train dataset for class 1 and class 2 in order 

