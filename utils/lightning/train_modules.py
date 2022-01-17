# delete for notebook
from utils.lightning.petfinder_dataset_v1 import PetFinderDataset, PetFinderDataModule
from utils.common.project_paths import GetPaths

import numpy as np
import torch
import pandas as pd
import yaml
from torch.utils.data import DataLoader
import torch.multiprocessing
from pytorch_lightning import loggers as pl_loggers

from sklearn.model_selection import StratifiedKFold

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import torch.nn.functional as F
from box import Box

import warnings
warnings.simplefilter('ignore')

with open(GetPaths().get_project_root('config', 'config_swin_v1.yaml')) as f:
    config = yaml.full_load(f)
config = Box(config)


class LightningTrainModule:
    def __init__(self):
        self.paths = GetPaths()

    def load_csv(self):
        train_csv = pd.read_csv(self.paths.get_data_folder('train.csv'))
        test_csv = pd.read_csv(self.paths.get_data_folder('test.csv'))
        sample_submit = pd.read_csv(self.paths.get_data_folder('sample_submission.csv'))
        cutmix_csv = pd.read_csv(self.paths.get_data_folder('cutmix_train.csv'))
        return train_csv, test_csv, sample_submit, cutmix_csv

    def bins(self, csv):
        num_bins = int(np.floor(1 + (3.3) * (np.log10(len(csv)))))
        # num_bins = int(np.ceil(2*((len(csv))**(1./3))))
        csv['bins'] = pd.cut(csv['Pawpularity'], bins=num_bins, labels=False)
        return csv

    def logging(self):
        pass