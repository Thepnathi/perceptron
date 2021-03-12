# Part 3 - Use the binary perceptron to train classifiers to discriminate between classes

import numpy as np
from DatasetLoader import DatasetLoader
from Perceptron import Perceptron
# https://numpy.org/doc/stable/reference/generated/numpy.sign.html

class PerceptronClassification:
    def compute_activation(self, dataRow, weight, bias):
        return np.dot(dataRow, weight) + bias

    def predict(self, dataRow, bias, weight):
        activation = self.compute_activation(dataRow, weight, bias)
        return np.sign(activation)

    def predict_test_dataset(self, test_data, bias, weight):
        correct_predict = 0
        for row in test_data:
            result = self.predict(row[:-1], bias, weight)
            correct_predict += 1 if result > 0 else 0
        accuracy = (correct_predict / len(test_data)) * 100
        print(f'Accuracy - {accuracy}%')

if __name__ == "__main__":
    # Loads the randomised dataset with two classes or all
    dataloader = DatasetLoader()
    # Data between class 1 and class 2
    train_data1 = dataloader.extract_two_classes_from_dataset('train.data', 1, 2)
    test_data1 = dataloader.extract_two_classes_from_dataset('test.data', 1, 2)
    # Data between class 2 and class 3
    train_data2 = dataloader.extract_two_classes_from_dataset('train.data', 2, 3)
    test_data2 = dataloader.extract_two_classes_from_dataset('test.data', 2, 3)
    # Data between class 1 and class 3
    train_data3 = dataloader.extract_two_classes_from_dataset('train.data', 1, 3)
    test_data3 = dataloader.extract_two_classes_from_dataset('test.data', 1, 3)

    # Initialise perceptron with start weight of 0 and bias of 0
    perceptron = Perceptron(0, 0)

    # Training the perceptron with 20 iterations
    bias1, weight1 = perceptron.train_perceptron(train_data1, 20)
    # bias2, weight2 = perceptron.train_perceptron(train_data2, 20)
    # bias3, weight3 = perceptron.train_perceptron(train_data3, 20)

    # Initialise classification class to compute accuracy of test data
    pred = PerceptronClassification()

    pred.predict(test_data1, bias1, weight1)
    # pred.predict(test_data2, bias2, weight2)
    # pred.predict(test_data3, bias3, weight3)
