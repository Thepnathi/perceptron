import numpy as np
from collections import Counter
from random import shuffle
from Constant import Constant

class DatasetHandler:
    def __init__(self, path=Constant.DATASET_DIR):
        self.path = path

    def load_dataset(self, fileName: str):
        openedFile = open(f'{self.path}/{fileName}', "r")
        fileLines = openedFile.readlines()
        dataset = []

        for line in fileLines:
            line = line.strip(Constant.NEWLINE)
            split_data = line.split(Constant.DATASET_DELIMETER)
            coverted_class = self.convert_class_to_int(split_data)
            numpy_array = np.array(coverted_class, np.float)
            dataset.append(numpy_array)
        return dataset

    def convert_class_to_int(self, row):
        if row[-1] == Constant.CLASSES[1]: row[-1] = 1
        elif row[-1] == Constant.CLASSES[2]: row[-1] = 2
        elif row[-1] == Constant.CLASSES[3]: row[-1] = 3
        return row

    def extract_all_classes_from_dataset(self, fileName: str, randomise=True):
        dataset = self.load_dataset(fileName)
        shuffle(dataset) if randomise else None
        feature_dataset, label_dataset = [], []
        for row in dataset:
            label_dataset.append(row[-1])
            feature_dataset.append(row[:-1])
        return feature_dataset, label_dataset

    def extract_two_classes_from_dataset(self, fileName: str, classOne, classTwo, randomise=True):
        dataset = self.load_dataset(fileName)
        shuffle(dataset) if randomise else None
        feature_dataset = []
        label_dataset = []
        for row in dataset:
            if row[-1] == classOne:
                label_dataset.append(1)
                feature_dataset.append(row[:-1])
            elif row[-1] == classTwo:
                label_dataset.append(-1)
                feature_dataset.append(row[:-1])
        return feature_dataset, label_dataset

if __name__ == "__main__":
    dataloader = DatasetHandler()
    train_data, train_label = dataloader.extract_two_classes_from_dataset('train.data', 1, 2)
    test_data, test_label = dataloader.extract_two_classes_from_dataset('test.data', 1, 2)
    all_test_data, all_test_label = dataloader.extract_all_classes_from_dataset(fileName='test.data', randomise=False)
    
    # Testing if the dataset are loaded correctly
    for i in range(len(test_data)):
        print(f'features - {test_data[i]} label - {test_label[i]}')

    for i in range(len(all_test_data)):
        print(f'features - {all_test_data[i]} label - {all_test_label[i]}')