import sklearn.svm
import yaml

from utils.lightning.petfinder_dataset_v1 import PetFinderDataModule
from model.swin_v1 import ClassSwin
from utils.lightning.train_modules import LightningTrainModule
import pytorch_lightning as pl
from pytorch_lightning import callbacks
import torch
import pandas as pd
from utils.common.project_paths import GetPaths

import torch.multiprocessing
from pytorch_lightning import loggers as pl_loggers

from sklearn.model_selection import StratifiedKFold

import warnings
import gc
from box import Box
import glob

warnings.simplefilter('ignore')

# load config
with open(GetPaths().get_project_root('config', 'config_swin_v1.yaml')) as f:
    config = yaml.full_load(f)
config = Box(config)


def main(cutmix=False):
    ##########
    torch.multiprocessing.set_sharing_strategy('file_system')
    seed = config.train.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    ###########

    weights = sorted(glob.glob(GetPaths().get_project_root('result', 'swin_cutmix', '*.pth')))

    # Run the Kfolds training loop
    skf = StratifiedKFold(
        n_splits=config.train.n_folds, shuffle=True, random_state=config.train.fold_random_state
    )
    lt = LightningTrainModule()
    train_file, test_file, sample_file, cutmix_csv = lt.load_csv()
    tmp_id = cutmix_csv['Id'].tolist()
    no_jpg = [i.replace('.jpg', '') for i in tmp_id]
    cutmix_csv['Id'] = no_jpg
    train_file = lt.bins(train_file)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X=train_file['Id'], y=train_file['bins'])):
        if config.train.single_model is True and fold != config.train.single_fold:
            continue
        print(f"{'=' * 20} Fold: {fold} {'=' * 20}")
        # data frames
        valid_df = train_file.loc[valid_idx].reset_index(drop=True)
        train_df = train_file.drop(index=valid_idx, axis=0).reset_index(drop=True)
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        # logger
        lightning_loggers = pl_loggers.CSVLogger(save_dir=f"./result/{config.train.log_name}/",
                                                 name=f'fold{fold}'
                                                 )

        if cutmix:
            train_df = pd.concat([train_df, cutmix_csv], join='inner', ignore_index=True)

        # load modules
        data_module = PetFinderDataModule(train_df, valid_df, test_file, config)
        train = data_module.train_dataloader()
        valid = data_module.val_dataloader()
        del data_module

        # n_steps = len(train)  # n_steps 1cycle 스케쥴러
        model = ClassSwin(cfg=config)
        if config.train.load_model:
            # model_dict = model.state_dict()
            # pre_trained_dict = torch.load(weights[fold])['state_dict']
            pre_trained_dict = torch.load(weights[fold])

            # new_dict = {}
            # for k, v in zip(model_dict.keys(), pre_trained_dict.values()):
            #     new_dict[k] = v

            model.load_state_dict(pre_trained_dict, strict=False)

        lr_monitor = callbacks.LearningRateMonitor()
        early_stopping = callbacks.EarlyStopping(monitor='valid_rmse', verbose=True, patience=config.early_stopper.patience)
        loss_checkpoint = callbacks.ModelCheckpoint(
            dirpath=f'./result/{config.train.log_name}/',
            filename='best_loss',
            monitor='valid_rmse',
            save_top_k=1,
            mode='min',
            save_last=False,
            verbose=True
        )

        trainer = pl.Trainer(
            max_epochs=config.train.epochs,
            gpus=config.train.n_gpus,
            strategy="ddp_spawn",
            callbacks=[lr_monitor, loss_checkpoint, early_stopping],
            logger=lightning_loggers,
            precision=config.train.precision,
        )

        trainer.fit(model, train, valid)

        # 메모리 청소
        del model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main(cutmix=config.train.cutmix)
