from torch.utils import data
import numpy as np
from PIL import Image
import pandas as pd
import os


class PetDataset(data.Dataset):
    def __init__(self, data_transform=None, data_csv=None):
        self.transform = data_transform
        self.file_path = os.getcwd()
        if data_csv is None:
            data_csv = pd.read_csv("petfinder-pawpularity-score/train.csv")
        self.label_df = data_csv

    def __getitem__(self, index):
        label = self.label_df.iloc[index, -1]/100
        img_id = self.label_df.iloc[index, 0]
        img_path = "petfinder-pawpularity-score/train/" + img_id + ".jpg"
        img = np.array(Image.open(img_path))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return self.label_df.shape[0]
