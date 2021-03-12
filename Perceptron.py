# Task 2 - Implement a binary perceptron
# Task 3 - Use the binary perceptron to train classifiers:

import numpy as np
from DatasetLoader import DatasetLoader

class Perceptron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def train_perceptron(self, train_data, num_epoch):
        weight = [0 for i in range(len(train_data[0])-1)]        # -1 because we do not want to include the last index
        bias = 0                                                   # which is the integer for the label
        for i in range(num_epoch):
            # print("Epoch number ", i)
            total = 0
            for row in train_data:
                activation = self.compute_activation(row[:-1], weight, bias)
                label = row[-1]
                if label * activation <= 0:    
                    total += 1    
                    weight = self.compute_new_weight(weight, row[:-1], label)          # Update weights and bias 
                    bias += label
            # print(f'Total number of new weight abjustment is {total}')
        return bias, weight

    def compute_activation(self, dataRow, weight, bias):
        return np.dot(weight, dataRow) + bias

    def compute_new_weight(self, oldWeight, row, class_value):
        newRow = np.multiply(class_value, row)
        return np.add(oldWeight, newRow)

if __name__ == "__main__":
    # Loads the randomised dataset with two classes or all
    dataloader = DatasetLoader()
    train_data = dataloader.extract_two_classes_from_dataset('train.data', 2, 3)
    test_data = dataloader.extract_two_classes_from_dataset('test.data', 2, 3)
    perceptron = Perceptron(0, 0)
    
    bias, weight = perceptron.train_perceptron(train_data, 20)
    print(len(train_data))

    # for i in range(len(test_data)):
    #     print(test_data[i])