from pathlib import Path
from typing import Dict, Any, Tuple, Union, List
import os

path_type = Union[str, os.PathLike, Path]
TRAIN_FILE = 'train.csv'

def setup_eval_period(sample_level: int = 0) -> List[Tuple[str, str]]:
    # sample level for only 2021 data
    if sample_level == 1:
        return [['2021-06-13',
                 '2021-09-13']]
    # one day worth of valid set
    if sample_level == 2:
        return [['2021-09-12', '2021-09-13']]
    
    # main level
    return  [['2020-01-01', '2020-04-01'],
             ['2020-09-01', '2021-01-01'],
             ['2021-01-01', '2021-04-01'],
             ['2021-04-01', '2021-07-01']]


def setup_dir(on_kaggle: bool = True, sample_level: int = 0) -> Tuple[path_type, path_type]:
    if on_kaggle:
        data_dir = Path('../input/g-research-crypto-forecasting/')
        raw_train_dir = (Path('../input/create-sample-dataset/data/raw/')
                         if sample_level > 0 else data_dir) 
    else:
        data_dir = raw_train_dir = Path('data/raw')
    
    if sample_level > 0:
        raw_train_dir = raw_train_dir.joinpath('sample', str(sample_level))
    
    return data_dir, raw_train_dir


def setup_env(on_kaggle: bool = True,
              sample_level: int = 0) -> Dict[str, Any]:
    data_dir, raw_train_dir = setup_dir(on_kaggle=on_kaggle, sample_level=sample_level)
    cv_period = setup_eval_period(sample_level=sample_level)
    config = {'dirs': {'data_dir': data_dir,
                       'raw_train_path': os.path.join(raw_train_dir, TRAIN_FILE)},
              'cv_period': cv_period}
    return config
