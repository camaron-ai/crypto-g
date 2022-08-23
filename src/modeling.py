from typing import Any, Callable, Dict, List
import pandas as pd
import numpy as np
from sklearn.model_selection._split import _BaseKFold
import time
from collections import defaultdict

config_type = Dict[str, Any]


def filter_by_index(df: pd.DataFrame, index) -> pd.DataFrame:
    return df.loc[index, :].reset_index(drop=True)


KEEP_COLS = ['Asset_ID', 'Asset_Name', 'time', 'timestamp', 'Target']


class Evaluator:
    def __init__(self,
                 cv: _BaseKFold,
                 fi_fn: Callable,
                 ):
        self.cv = cv
        self.fi_fn = fi_fn

    def run(self, train_fn: Callable,
            config: config_type,
            data: pd.DataFrame,
            ) -> List[Dict[str, Any]]:
        output = defaultdict(list)

        for fold, (train_idx, valid_idx) in enumerate(self.cv.split(data)):
            print(f'fold={fold}')
            train_data = data.loc[train_idx, :]
            valid_data = data.loc[valid_idx, :]
            start_time = time.time()
            model = train_fn(config=config,
                             train_data=train_data,
                             valid_data=valid_data)
            elapsed_time = (time.time() - start_time) / 60
            print(f'elapsed time: {elapsed_time:.4f}')
            output['model'].append(model)
            prediction = valid_data.loc[:, KEEP_COLS]
            prediction['yhat'] = model.predict(valid_data)
            prediction['fold'] = fold
            output['prediction'].append(prediction)

            if self.fi_fn:
                fi = self.fi_fn(config, model, valid_data)
                output['fi'].append(fi)
        return output



