import numpy as np
from sklearn.model_selection._split import _BaseKFold
from typing import List, Tuple, Iterable
import pandas as pd
from datetime import datetime



def get_date_range(dates: pd.Series):
    return dates.agg(('min', 'max'))


class TimeSeriesSplit(_BaseKFold):
    def __init__(self, periods: List[Tuple[str, str]],
                 train_days: int = None,
                 gap: int = 1,
                 gap_unit: int = 'd',
                 dt_col: str = 'date'):
        self.dt_col = dt_col
        self.periods = periods
        self.train_days = train_days
        self.gap = gap
        self.gap_unit = gap_unit
        
    def __len__(self) -> int:
        return len(self.periods)
    
    def check_input(self, X: pd.DataFrame, y=None, groups=None) -> None:
        assert self.dt_col in X.columns, f'{self.dt_col} do not exits in input dataframe'
        
    def split(self, X: pd.DataFrame, y=None, groups=None) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        dates = X[self.dt_col]
        self.check_input(X)
        
        first_date = dates.min()
        
        indices = np.arange(len(X))
        for period in self.periods:
            first_valid_date = pd.to_datetime(period[0])
            
            last_train_date = first_valid_date - pd.to_timedelta(self.gap, unit=self.gap_unit)
            
            if self.train_days:
                first_train_date = last_train_date - pd.to_timedelta(self.train_days, unit='d')
                first_train_date = np.maximum(first_train_date, first_date)
            else:
                first_train_date = first_date
            
            valid_mask = dates.between(*period)
            train_mask = (dates.between(first_train_date, last_train_date)) & (dates < first_valid_date)
            
            yield indices[train_mask], indices[valid_mask]



def gen_eval_periods(start_date: str,
                     n_test: int,
                     n_splits: int,
                     unit: str = 'd') -> List[Tuple[datetime, datetime]]:
    start_date = pd.to_datetime(start_date)
    eval_periods = []
    for _ in range(n_splits):
        end_date = start_date + pd.to_timedelta(n_test, unit=unit)
        eval_periods.append([start_date, end_date])
        start_date = end_date + pd.to_timedelta(1, unit=unit)
    return eval_periods
    
    
    