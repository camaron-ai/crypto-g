import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Dict, Any
from src.pipeline.transforms import TMF_DISPATCHER


def timestamps_to_timestep(timestamps: np.ndarray,
                           step_size: int = 60 * 24) -> np.ndarray:
    timestep = (timestamps.max() - timestamps) // 60
    timestep = (timestep / step_size)
    return timestep


def compute_exp_time_weight(timestamps: np.ndarray,
                            alpha: float,
                            step_size: int = 24 * 60) -> np.ndarray:
    timesteps = timestamps_to_timestep(timestamps, step_size=step_size)
    return np.power(alpha, timesteps)



def get_train_test_tuple(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = (df.loc[df.model_group == "train"]
             .drop("model_group", axis=1)
             .reset_index(drop=True))
    test = (df.loc[df.model_group == "test"]
            .drop("model_group", axis=1)
            .reset_index(drop=True))
    return train, test



def get_Xy_tuple(data: pd.DataFrame,
                 target: List[str]) -> Tuple[np.ndarray, np.ndarray]:

    features = data.filter(regex='ft_')
    return features.to_numpy(), data.loc[:, target].to_numpy()



KEEP_COLS = ['Asset_ID', 'Asset_Name', 'time', 'timestamp', 'Target']
def run_experiment(config,
                   train_model_fn: Callable,
                   data: pd.DataFrame) -> Dict[str, Any]:    
    # build model
    train_data, valid_data = get_train_test_tuple(data)

    train_features, train_target = get_Xy_tuple(train_data,
                                                        target=config['target'])
    valid_features, valid_target = get_Xy_tuple(valid_data,
                                                        target=config['target'])

    assert train_features.shape[1] == valid_features.shape[1]

    if config['training']['exp_time_weight'] is not None:
        train_sample_weight = compute_exp_time_weight(train_data['timestamp'], config['training']['exp_time_weight'])
    else:
        train_sample_weight = None
        
        
    use_target_tmf = config['training']['target_tmf']
    if use_target_tmf:
        target_tmf = TMF_DISPATCHER.get(config['training']['target_tmf'])
        train_target = target_tmf.transform(train_target)
        valid_target = target_tmf.transform(valid_target)
    
    model = train_model_fn(config,
                           train_features=train_features,
                           train_target=train_target,
                           train_sample_weight=train_sample_weight,
                           valid_features=valid_features,
                           valid_target=valid_target,
                           valid_sample_weight=None)

    valid_yhat = model.predict(valid_features)
    if use_target_tmf:
        valid_yhat = target_tmf.inverse_transform(valid_yhat)
    
    pd_inference = valid_data.loc[:, KEEP_COLS]
    pd_inference['yhat'] = valid_yhat
    
    return {'inference': pd_inference, 'model': model}
