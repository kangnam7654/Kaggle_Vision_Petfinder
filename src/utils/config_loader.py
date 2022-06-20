import yaml

def config_load(config_path):
    with open(config_path) as f:
        config = yaml.full_load(f)
    return config