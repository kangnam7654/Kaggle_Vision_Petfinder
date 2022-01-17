from model.swin_v1 import ClassSwin
import torch
from utils.common.project_paths import GetPaths
import yaml
import glob
from box import Box

def load_config():
    paths = GetPaths()
    config_list = glob.glob(paths.get_project_root('config', '*.yaml'), recursive=True)
    with open(config_list[0]) as f:
        config = yaml.full_load(f)
    config = Box(config)
    return config


def load_weights():
    paths = GetPaths()
    weights = sorted(glob.glob(paths.get_project_root('result', 'cutmix_2layer', '*.ckpt'), recursive=True))
    return weights

def main():
    config = load_config()
    weights = load_weights()
    for weight in weights:
        model = ClassSwin(cfg=config)
        state = torch.load(weight, map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state)
        weight_name_split = weight.split('.')
        weight_rename = f'{weight_name_split[0]}_state.pth'
        torch.save(model.state_dict(), weight_rename)

if __name__ == '__main__':
    main()
