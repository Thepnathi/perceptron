import numpy as np
from Constant import Constant

class DataLoader:
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

if __name__ == "__main__":
    dataloader = DataFile()
    train_data = dataloader.load_dataset('train.data')
    test_data = dataloader.load_dataset('test.data')

    for i in range(len(test_data)):
        print(f'{i+1} - {test_data[i]}')