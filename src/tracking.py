import mlflow
import os
from typing import Dict, Any, Union
from datetime import datetime as dt
from omegaconf import OmegaConf
import omegaconf


DATE_FORMAT: str = '%Y-%m-%d-%H-%M-%S'

def str_now():
    return dt.now().strftime(DATE_FORMAT)


def allign_dictionary_subclasses(dictionary: Dict[str, Union[Any, Dict[str, Any]]]):
    output = {}
    for item, value in dictionary.items():
        if isinstance(value, dict):
            _subdict = allign_dictionary_subclasses(value)
            renamed_subdict = {f'{item}__{subitem}': subvalue for subitem, subvalue in _subdict.items()}
            output.update(renamed_subdict)
        else:
            output[f'{item}'] = value
    return output


def track_experiment(run_name: str,
                     mlflow_experiment: str,
                     config: Dict[str, Any],
                     scores: Dict[str, float],
                     artifacts_dir: str = None,
                     ):
    mlflow.set_experiment(mlflow_experiment)
    if isinstance(config, omegaconf.DictConfig):
        config = OmegaConf.to_container(config)

    parameters = allign_dictionary_subclasses(config)
    seed = os.environ.get('SEED', 'Not Set')

    tags = {'seed': seed}
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(tags)
        # log parameters
        mlflow.log_params(parameters)
        # log metrics
        mlflow.log_metrics(scores)
        # log all artifacts
        if artifacts_dir:
            mlflow.log_artifacts(artifacts_dir)