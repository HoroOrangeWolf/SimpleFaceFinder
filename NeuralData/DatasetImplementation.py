import numpy
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class DatasetImplementation(Dataset):
    def __init__(self, data_path, picture_size=(1028, 1028), picture_transformation=None):
        self.data_file = pd.read_csv(data_path + '/data.csv')
        self.length = len(self.data_file)
        self.picture_transformation = picture_transformation
        self.data_path = data_path
        self.picture_size = picture_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        y = self.data_file.iloc[idx, 0:-1]
        x = self.data_file.iloc[idx, -1]
        x = Image.open(self.data_path + '/' + x).convert('L')

        if self.picture_transformation:
            x = self.picture_transformation(x)

        value = [y[0] / self.picture_size[0], y[1] / self.picture_size[1], y[2] / self.picture_size[0],
                 y[3] / self.picture_size[1]]

        return x, numpy.array(value)
