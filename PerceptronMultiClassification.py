import numpy as np
from DatasetHandler import DatasetHandler
from Perceptron import Perceptron
from PerceptronClassification import PerceptronClassification
from Constant import view_dataset 


class PerceptronMultiClassification(PerceptronClassification):
    def activation_score_raw(self, data_row, weight, bias) -> int:
        activation = np.dot(data_row, weight) + bias
        return activation

    def compute_multiclass_prediction_accuracy(self, feature_dataset, label_dataset, *trained_models):
        correct_prediction =  0

        for i, data_row in enumerate(feature_dataset):
            confidence_scores = []
            for model in trained_models:
                activation_score = self.activation_score_raw(data_row, model.get_weight(), model.get_bias())
                confidence_scores.append(activation_score)

            selected_model = np.argmax(confidence_scores)
            prediction = self.activation_score(data_row, trained_models[selected_model].get_weight(), trained_models[selected_model].get_bias())

            correct_prediction += 1 if prediction == label_dataset[i] else correct_prediction

        return (correct_prediction / len(feature_dataset)) * 100


if __name__ == "__main__":
    # Load all train dataset. Each will have one class as positive and the rest negative
    # The training dataset with one class positive will be used for one-vs-rest perceptron classification. There will be k dataset for k classes
    train1 = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=False).make_one_class_positive(1)
    train2 = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=False).make_one_class_positive(2)
    train3 = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=False).make_one_class_positive(3)

    # Load the test dataset that will be used for classification accuracy
    testAll = DatasetHandler().extract_all_classes_from_dataset('test.data', randomise=False)

    onlyClass1_x, onlyClass1_y = testAll.feature_dataset[:10], testAll.label_dataset[:10]

    view_dataset(onlyClass1_x, onlyClass1_y)

    # Initialise the perceptron algorithm to train our model
    perceptron = Perceptron(0, 0)

    # Training the model
    model1 = perceptron.train_perceptron(train1.feature_dataset, train1.label_dataset, 20)
    model2 = perceptron.train_perceptron(train2.feature_dataset, train2.label_dataset, 20)
    model3 = perceptron.train_perceptron(train3.feature_dataset, train3.label_dataset, 20)
