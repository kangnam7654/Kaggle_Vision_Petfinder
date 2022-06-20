import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

import pytorch_lightning as pl
from pytorch_lightning import callbacks, loggers
from pytorch_lightning import seed_everything

from model.swin import SwinModel

from utils.project_paths import GetPaths
from dataload.petfinder_dataloader import PetFinderDataModule
from dataload.petfinder_transform import load_transforms
from utils.make_bins import make_bins
from utils.config_loader import config_load


import warnings
import gc

warnings.simplefilter("ignore")

# load config
config_path = GetPaths.get_project_root("config", "config.yaml")
config = config_load(config_path)

def main():
    seed_everything(config['train']["seed"])

    # load_csv
    train_file_path = GetPaths.get_data_folder('train.csv')
    train_file = pd.read_csv(train_file_path)

    # Run the Kfolds training loop
    skf = StratifiedKFold(**config["train"]["skf"])
    train_file = make_bins(train_file)  # 점수를 계층형으로 변환

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(X=train_file["Id"], y=train_file["bins"])
    ):  # Cross validation

        print(f"{'=' * 20} Fold: {fold} {'=' * 20}")
        # data frames
        train_df = train_file.loc[train_idx].reset_index(drop=True)
        valid_df = train_file.loc[valid_idx].reset_index(drop=True)

        # load modules
        transform = load_transforms(config=config)
        data_module = PetFinderDataModule(train_df=train_df, valid_df=valid_df, cfg=config, transform=transform)
        train = data_module.train_dataloader()
        valid = data_module.val_dataloader()

        # n_steps = len(train)  # n_steps 1cycle 스케쥴러
        model = SwinModel(cfg=config)

        lr_monitor = callbacks.LearningRateMonitor(logging_interval='step')  # lr monitor
        early_stopping = callbacks.EarlyStopping(**config['train']['early_stopping'])  # early stop
        wandb_logger = loggers.WandbLogger(**config['train']['wandb_logger'])  # logger
        checkpoint = callbacks.ModelCheckpoint(**config['train']['checkpoint']) # checkpoint save

        trainer = pl.Trainer(
            callbacks=[lr_monitor, checkpoint, early_stopping],
            logger=wandb_logger,
            **config['train']['trainer']
        )

        trainer.fit(model, train, valid)

        # 메모리 청소
        del model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
