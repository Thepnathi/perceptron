# Part 3 - Use the binary perceptron to train classifiers to discriminate between classes

import numpy as np
from DatasetHandler import DatasetHandler
from Perceptron import Perceptron

class PerceptronClassification:
    def predictict(self, dataRow, bias, weight):
        activation = np.dot(dataRow, weight) + bias
        return np.sign(activation)

    # Given a test dataset with feature + label array. Returns the total accuracy for correct prediction
    def compute_test_dataset_accuracy(self, test_data, test_label, bias, weight):
        correct_predictict = 0
        for i, row in enumerate(test_data):
            activation_score = self.predictict(row, bias, weight)
            if activation_score == test_label[i]:
                correct_predictict += 1
        return (correct_predictict / len(test_data)) * 100

if __name__ == "__main__":
    # Loads the randomised dataset with two classes or all

     # Test and Train Dataset for class 1 and class 2
    train1 = DatasetHandler().extract_two_classes_from_dataset('train.data', 1, 2, randomise=False)
    test1 = DatasetHandler().extract_two_classes_from_dataset('test.data', 1, 2, randomise=False)

    # Test and Train Dataset for class 2 and class 3
    train2 = DatasetHandler().extract_two_classes_from_dataset('train.data', 2, 3, randomise=False)
    test2 = DatasetHandler().extract_two_classes_from_dataset('test.data', 2, 3, randomise=False)

    # Test and Train Dataset between class 1 and class 3
    train3 = DatasetHandler().extract_two_classes_from_dataset('train.data', 1, 3, randomise=False)
    test3 = DatasetHandler().extract_two_classes_from_dataset('test.data', 1, 3, randomise=False)

    # Initialise perceptron algorithm with start weight of 0 and bias of 0
    perceptron = Perceptron(0, 0)

    # Training the perceptron with 20 iterations for the three types of train dataset we have loaded
    bias1, weight1 = perceptron.train_perceptron(train1.feature_dataset, train1.label_dataset, 20)
    bias2, weight2 = perceptron.train_perceptron(train2.feature_dataset, train2.label_dataset, 20)
    bias3, weight3 = perceptron.train_perceptron(train3.feature_dataset, train3.label_dataset, 20)

    # Initialise classification class to compute accuracy of test data
    predict = PerceptronClassification()

    acc1 = predict.compute_test_dataset_accuracy(test1.feature_dataset, test1.label_dataset, bias1, weight1)
    acc2 = predict.compute_test_dataset_accuracy(test2.feature_dataset, test2.label_dataset, bias2, weight2)
    acc3 = predict.compute_test_dataset_accuracy(test3.feature_dataset, test3.label_dataset, bias3, weight3)

    print(f"Test dataset with class 1 and 2 with accuracy of {acc1}%")
    print(f"Test dataset with class 2 and 3 with accuracy of {acc2}%")
    print(f"Test dataset with class 1 and 3 with accuracy of {acc3}%")