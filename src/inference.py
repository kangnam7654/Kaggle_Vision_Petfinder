import gc

from utils.project_paths import GetPaths
from utils.lightning.petfinder_dataset_v1 import PetFinderDataModule, get_default_transforms
from model.swin_v1 import ClassSwin
import pandas as pd
import glob
import pytorch_lightning as pl
import torch
import yaml
from box import Box

def load_csv():
    train_csv = pd.read_csv(GetPaths().get_data_folder('train.csv'))
    test_csv = pd.read_csv(GetPaths().get_data_folder('test.csv'))
    sample_submit = None
    return train_csv, test_csv, sample_submit

def load_config():
    paths = GetPaths()
    config_list = glob.glob(paths.get_project_root('config', '*.yaml'), recursive=True)
    with open(config_list[0]) as f:
        config = yaml.full_load(f)
    config = Box(config)
    return config


def load_weights():
    paths = GetPaths()
    weights = sorted(glob.glob(paths.get_project_root('result', 'swin_transfer6', 'best_loss.ckpt'), recursive=True))
    return weights


def main():
    # config load
    config = load_config()
    train_file, test_file, sample_file = load_csv()
    weights = load_weights()

    data_module = PetFinderDataModule(train_df=train_file, test_df=test_file, cfg=config)
    train = data_module.train_dataloader()
    test = data_module.test_dataloader()
    predict = data_module.predict_dataloader()

    # all init args were saved to the checkpoint
    all_preds = []
    for weight in weights:
        model = ClassSwin(cfg=config)
        state = torch.load(weight, map_location=torch.device('cpu'))['state_dict']
        # state = torch.load(weight)['state_dict']
        # state = torch.load(weight, map_location=torch.device('cpu'))
        model.load_state_dict(state, strict=True)
        weight_name_split = weight.split('.')
        weight_rename = f'{weight_name_split[0]}_.{weight_name_split[1]}'
        torch.save(model.state_dict(), weight_rename)
        break
        model.eval()
        model.freeze()
        trainer = pl.Trainer(gpus=1, precision=32)
        pred = trainer.predict(model=model, dataloaders=predict)
        all_preds.append(pred)

        # 메모리 청소
        del model
        torch.cuda.empty_cache()
        gc.collect()

    tensor_all_preds = torch.tensor(all_preds).float()
    prediction = torch.mean(tensor_all_preds, dim=0)
    submit = pd.DataFrame({
        'Id': train_file['Id'],
        'Pawpularity': prediction.squeeze().cpu().numpy(),
        'label': train_file['Pawpularity']
    })

    submit.to_csv('result_preds.csv')
    return submit

if __name__ == '__main__':
    submit = main()