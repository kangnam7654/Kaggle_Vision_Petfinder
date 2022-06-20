import pandas as pd
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.core import LightningDataModule
from torch.utils.data import DataLoader

from utils.config_loader import config_load
from dataload.petfinder_dataset import PetFinderDataset


class PetFinderDataModule(LightningDataModule):
    """Data module of Petfinder profiles."""
    def __init__(self, train_df=None, valid_df=None, test_df=None, cfg=None, transform=None):
        super().__init__()
        self._train_df = train_df
        self._valid_df = valid_df
        self._test_df = test_df
        self._cfg = cfg
        self.transform = transform

    def __create_dataset(self, mode="train"):
        if mode == "train":
            return PetFinderDataset(
                self._train_df, train=True, transform=self.transform["train"]
            )
        elif mode == "valid":
            return PetFinderDataset(
                self._valid_df, train=True, transform=self.transform["valid"]
            )
        elif mode == "predict":
            return PetFinderDataset(
                self._test_df, train=False, transform=self.transform["test"]
            )
        else:
            raise Exception('mode should be in [train, valid, predict]')

    def train_dataloader(self):
        dataset = self.__create_dataset("train")
        return DataLoader(dataset, **self._cfg['train_loader'])

    def val_dataloader(self):
        dataset = self.__create_dataset("valid")
        return DataLoader(dataset, **self._cfg['valid_loader'])

    def predict_dataloader(self):
        dataset = self.__create_dataset("predict")
        return DataLoader(dataset, **self._cfg['test_loader'])
