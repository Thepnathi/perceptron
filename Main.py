from DatasetHandler import DatasetHandler
from Perceptron import Perceptron
from PerceptronClassification import PerceptronClassification

if __name__ == "__main__":
    # Load all train dataset. Each will have one class as positive and the rest negative
    # The training dataset with one class positive will be used for one-vs-rest perceptron classification. There will be k dataset for k classes
    train1 = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=False).make_one_class_positive(1)
    train2 = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=False).make_one_class_positive(2)
    train3 = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=False).make_one_class_positive(3)

    # Load the test dataset that will be used for classification accuracy
    testAll = DatasetHandler().extract_all_classes_from_dataset('test.data', randomise=False)

    # Initialise the perceptron algorithm to train our model
    perceptron = Perceptron(0, 0)

    # Training the model
    model1 = perceptron.train_perceptron(train1.feature_dataset, train1.label_dataset, 20)
    model2 = perceptron.train_perceptron(train2.feature_dataset, train2.label_dataset, 20)
    model3 = perceptron.train_perceptron(train3.feature_dataset, train3.label_dataset, 20)


    def testDataset(feature, label):
        for i in range(len(feature)):
            print(f'features - {feature[i]}, label - {label[i]}')

    # testDataset(train1.feature_dataset, train1.label_dataset)
    # testDataset(train2.feature_dataset, train2.label_dataset)
    # testDataset(train3.feature_dataset, train3.label_dataset)
    testDataset(testAll.feature_dataset, testAll.label_dataset)