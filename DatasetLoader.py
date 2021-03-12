import numpy as np
from collections import Counter
from random import shuffle
from Constant import Constant

class DatasetLoader:
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

    def extract_two_classes_from_dataset(self, fileName: str, classOne=None, classTwo=None):
        dataset = self.load_dataset(fileName)
        shuffle(dataset)
        if classOne and classTwo:
            binary_dataset = []
            for row in dataset:
                if row[-1] == classOne:
                    row[-1] = 0
                    binary_dataset.append(row)
                elif row[-1] == classTwo:
                    row[-1] = 1
                    binary_dataset.append(row)
            return binary_dataset
        return dataset

if __name__ == "__main__":
    dataloader = DatasetLoader()
    train_data = dataloader.extract_two_classes_from_dataset('train.data')
    test_data = dataloader.extract_two_classes_from_dataset('test.data', 1, 2)

    for i in range(len(test_data)):
        print(f'{i+1} - {test_data[i]}')