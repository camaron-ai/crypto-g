from typing import Union, Dict, Any
import yaml
import os
from omegaconf import OmegaConf
import omegaconf
from pathlib import Path

# TYPES
pathtype = Union[str, os.PathLike]
config_type = Dict[str, Any]

# CONFIG FILE
def load_yml(config_file: pathtype) -> config_type:
    """Helper function to read a yaml file"""
    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config

def load_config(config_file: pathtype) -> omegaconf.dictconfig.DictConfig:
    """Helper function to read a config file"""

    if not os.path.exists(config_file):
        raise FileNotFoundError(f'{config_file} file do not exists')

    config = load_yml(config_file)
    return config


class Dispatcher(dict):
    def get(self, config: Dict[str, Any]):
        instance = config['instance']
        parameters = config.get('parameters', {})
        return self[instance](**parameters)
