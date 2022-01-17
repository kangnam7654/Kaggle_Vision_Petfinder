# delete this for notebook
# others
import os

import albumentations as A
import cv2
import pandas as pd
import yaml
from albumentations.pytorch import ToTensorV2
from box import Box
from pytorch_lightning.core import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from utils.common.project_paths import GetPaths

# config load
with open(GetPaths().get_project_root('config', 'config_swin_v1.yaml')) as f:
    config = yaml.full_load(f)
config = Box(config)

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB

transform = {
    "train": A.Compose(
        [A.HorizontalFlip(),
         A.VerticalFlip(),
         A.RandomRotate90(),
         A.Downscale(scale_min=0.5, scale_max=0.99, p=1),
         A.ColorJitter(p=0.8),
         # A.ToGray(p=1),
         A.CoarseDropout(),
         A.LongestMaxSize(max_size=config.transform.image_size),
         A.PadIfNeeded(min_height=config.transform.image_size, min_width=config.transform.image_size,
                       border_mode=cv2.BORDER_CONSTANT, p=0.5),
         A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
         A.Resize(384, 384),
         ToTensorV2()
         ]),

    "valid": A.Compose(
        [
            # A.ToGray(p=1),
            A.LongestMaxSize(max_size=config.transform.image_size),
            A.PadIfNeeded(min_height=config.transform.image_size, min_width=config.transform.image_size,
                          border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            A.Resize(384, 384),
            ToTensorV2()
        ]),

    'test': A.Compose(
        [
            # A.ToGray(p=1),
            A.LongestMaxSize(max_size=config.transform.image_size),
            A.PadIfNeeded(min_height=config.transform.image_size, min_width=config.transform.image_size,
                          border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            A.Resize(384, 384),
            ToTensorV2()
        ])
}
def get_default_transforms():
    return transform

def get_tta_transforms():
    transform =[
        A.Compose(
            [
                A.HorizontalFlip(always_apply=True),
                A.LongestMaxSize(max_size=config.transform.image_size),
                A.PadIfNeeded(min_height=config.transform.image_size, min_width=config.transform.image_size,
                              border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2()
            ]
        ),
        A.Compose(
            [
                A.VerticalFlip(always_apply=True),
                A.LongestMaxSize(max_size=config.transform.image_size),
                A.PadIfNeeded(min_height=config.transform.image_size, min_width=config.transform.image_size,
                              border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2()
            ]
        ),
        A.Compose(
            [
                A.HorizontalFlip(always_apply=True),
                A.VerticalFlip(always_apply=True),
                A.LongestMaxSize(max_size=config.transform.image_size),
                A.PadIfNeeded(min_height=config.transform.image_size, min_width=config.transform.image_size,
                              border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2()
            ]
        )
    ]

    return transform

class PetFinderDataset(Dataset):
    def __init__(self, df, train=True, predict=False, transform=None, tta=None):
        # delete for notebook
        self.paths = GetPaths()
        self.train = train
        self.df = self.pre_df(df)
        self.x = self.df['Id'].values
        self.transform = transform
        self.tta = tta
        self.predict = predict

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # edit this in notebook
        if self.train or self.predict:
            image_dir = self.paths.get_data_folder("cutmix")
        else:
            image_dir = self.paths.get_data_folder("test")


        form = f'{self.x[idx]}.jpg'
        image_path = os.path.join(image_dir, form)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None: # cutmix
            image_dir = self.paths.get_data_folder("cutmix_train")
            image_path = os.path.join(image_dir, form)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # transform
        if self.transform is not None:
            image_origin = self.transform(image=image)['image']
        else:
            image_origin = image

        if self.train:
            labels = self.df['norm_score'].values
            label = labels[idx]
            return image_origin, label
        else:
            tta_images = [image_origin]
            for tta in self.tta:
                tta_image = tta(image=image)['image']
                tta_images.append(tta_image)
            return tta_images

    def pre_df(self, df):
        if self.train:
            df['bins'] = pd.cut(df['Pawpularity'], bins=20, labels=False)
            df['bins_'] = pd.cut(df['Pawpularity'], bins=20)
            df['norm_score'] = df['Pawpularity'] / 100
            return df
        else:
            return df

class PetFinderDataModule(LightningDataModule):
    """Data module of Petfinder profiles."""
    def __init__(self, train_df=None, valid_df=None, test_df=None, cfg=None):
        super().__init__()
        self._train_df = train_df
        self._valid_df = valid_df
        self._test_df = test_df
        self._cfg = cfg
        self.trans = get_default_transforms()
        self.ttas = get_tta_transforms()

    def __create_dataset(self, mode='train'):
        if mode == 'train':
            return PetFinderDataset(self._train_df, train=True, transform=self.trans['train'])
        elif mode == 'valid':
            return PetFinderDataset(self._valid_df, train=True, transform=self.trans['valid'])
        elif mode == 'test':
            return PetFinderDataset(self._test_df, train=False, transform=self.trans['test'], tta=self.ttas)
        elif mode == 'predict':
            return PetFinderDataset(self._train_df, train=False, predict=True, transform=self.trans['test'], tta=self.ttas)

    def train_dataloader(self):
        dataset = self.__create_dataset('train')
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset('valid')
        return DataLoader(dataset, **self._cfg.valid_loader)

    def predict_dataloader(self):
        dataset = self.__create_dataset('predict')
        return DataLoader(dataset, **self._cfg.test_loader)

    def test_dataloader(self):
        dataset = self.__create_dataset('test')
        return DataLoader(dataset, **self._cfg.test_loader)




