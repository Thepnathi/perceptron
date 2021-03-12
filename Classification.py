# Part 3 - Use the binary perceptron to train classifiers to discriminate between classes

import numpy as np
from DatasetLoader import DatasetLoader
from Perceptron import Perceptron
# https://numpy.org/doc/stable/reference/generated/numpy.sign.html

class PerceptronClassification:
    def predict(self, dataRow, bias, weight):
        activation = np.dot(dataRow, weight) + bias
        return np.sign(activation)

    def predict_test_dataset(self, test_data, test_label, bias, weight):
        correct_predict = 0
        for i, row in enumerate(test_data):
            activation_score = self.predict(row, bias, weight)
            if activation_score == test_label[i]:
                correct_predict += 1
        accuracy = (correct_predict / len(test_data)) * 100
        print(f'Accuracy - {accuracy}%')

if __name__ == "__main__":
    # Loads the randomised dataset with two classes or all
    dataloader = DatasetLoader()
    # Data between class 1 and class 2
    train_data1, train_label1 = dataloader.extract_two_classes_from_dataset('train.data', 1, 2, randomise=True)
    test_data1, test_label1 = dataloader.extract_two_classes_from_dataset('test.data', 1, 2)
    # Data between class 2 and class 3
    train_data2, train_label2 = dataloader.extract_two_classes_from_dataset('train.data', 2, 3, randomise=True)
    test_data2, test_label2 = dataloader.extract_two_classes_from_dataset('test.data', 2, 3)
    # # Data between class 1 and class 3
    train_data3, train_label3 = dataloader.extract_two_classes_from_dataset('train.data', 1, 3, randomise=True)
    test_data3, test_label3 = dataloader.extract_two_classes_from_dataset('test.data', 1, 3)

    # Initialise perceptron with start weight of 0 and bias of 0
    perceptron = Perceptron(0, 0)

    # Training the perceptron with 20 iterations
    bias1, weight1 = perceptron.train_perceptron(train_data1, train_label1, 20)
    bias2, weight2 = perceptron.train_perceptron(train_data2, train_label2, 20)
    bias3, weight3 = perceptron.train_perceptron(train_data3, train_label2, 20)

    # Initialise classification class to compute accuracy of test data
    pred = PerceptronClassification()

    print("Test dataset with class 1 and 2")
    pred.predict_test_dataset(test_data1, test_label1, bias1, weight1)
    print("Test dataset with class 2 and 3")
    pred.predict_test_dataset(test_data2, test_label2, bias2, weight2)
    print("Test dataset with class 1 and 3")
    pred.predict_test_dataset(test_data3, test_label3, bias3, weight3)
