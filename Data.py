from Constant import Constant

class DataFile:
    def __init__(self, path=Constant.DATASET_DIR):
        self.path = path

    def readDataset(self, fileName: str):
        filePath = f'{self.path}/{fileName}'



if __name__ == "__main__":
    dataloader = DataFile()
    dataloader.readDataset('test.data')
    