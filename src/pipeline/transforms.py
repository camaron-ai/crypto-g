import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
import numpy as np
from pandas.api.types import is_numeric_dtype
from src.util import Dispatcher


def check_target(y):
    assert np.isnan(y).sum() == 0, 'target has nan'
    assert np.isinf(y).sum() == 0, 'target has inf'


def check_features(X: pd.DataFrame,
                   allow_nan: bool = False):
    for name, value in X.items():
        assert np.isinf(value).sum() == 0, f'{name} has inf values'
        assert allow_nan or np.isnan(value), f'{name} has NaNs'
        assert is_numeric_dtype(value), f'{name} is not numeric'


class FilterFeatures(BaseEstimator, TransformerMixin):
    def __init__(self,
                 features: List[str] = None,
                 sort: bool = False,
                  allow_nan: bool = True):
        self.sort = sort
        self.features = features[:]
        self.allow_nan = allow_nan
        if self.sort:
            self.features.sort()
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame):
        outX = X.loc[:, self.features]
        check_features(outX, allow_nan=self.allow_nan)
        return outX


class AsClassification:
    def transform(self, y: np.ndarray):
        
        profit = (y >= 0).astype(np.int64)
        if not hasattr(self, 'mean'):
            self.mean = profit.mean()
        return profit

    def inverse_transform(self, y: np.ndarray):
        return (y - self.mean).astype(np.float32)



TMF_DISPATCHER = Dispatcher()
TMF_DISPATCHER['cls'] = AsClassification