import os
import sys
from pathlib import Path

sys.path.append(Path(__file__).parents[1])

import cv2

from utils.project_paths import GetPaths
from torch.utils.data import Dataset


class PetFinderDataset(Dataset):
    def __init__(self, df, train=True, transform=None):
        self.train_image_dir = GetPaths.get_data_folder("train")
        self.train = train
        self.df = self.df_preprocess(df)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        indice = self.df.loc[idx]
        train_image_name = f'{indice["Id"]}.jpg'
        train_image_path = os.path.join(self.train_image_dir, train_image_name)
        image = self.cv2_readimg(train_image_path)

        if self.transform:
            image = self.transform(image=image)['image']

        if self.train:
            label = indice['norm_score']
            return image, label
        else:
            return image

    @staticmethod
    def cv2_readimg(img_path):
        return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    @staticmethod
    def df_preprocess(df):
        df["norm_score"] = df["Pawpularity"] / 100
        return df
