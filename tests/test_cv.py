from src import cv
import numpy as np
import pandas as pd
import pytest


TEST_CASES = [
    [None,
    1,
    'd',
    [['2021-04-01', '2021-04-30'],
     ['2021-06-16', '2021-06-25']],
    [['2021-01-01', '2021-03-31'],
     ['2021-01-01', '2021-06-15']]
    ],

    [30,
    3,
    'd',
    [['2021-05-15', '2021-06-30'],
     ['2021-07-16', '2021-12-25']],
    [['2021-04-12', '2021-05-12'],
     ['2021-06-13', '2021-07-13']]
    ]
    ,
    [None,
    30,
    'min',
    [['2021-05-15', '2021-06-30'],
     ['2021-07-16', '2021-12-25']],
    [['2021-01-01', '2021-05-14 11:30:00pm'],
     ['2021-01-01', '2021-07-15 11:30:00pm']]
    ],

    [266,
    1,
    'h',
    [['2021-11-10', '2021-12-30']],
    [['2021-02-16 11:00:00pm', '2021-11-09 11:00:00pm']],
    ]
]

def check_date(a: str, b: str):
    a = pd.to_datetime(a)
    b = pd.to_datetime(b)
    assert a == b, f'{a} != {b}'


@pytest.mark.parametrize('train_days, gap, gap_unit, eval_periods, train_periods', TEST_CASES)
def test_time_series_split(train_days: int, gap: int, gap_unit: str, eval_periods, train_periods):
    # one year
    dates = pd.DataFrame({'date': pd.date_range('2021-01-01', '2021-12-31', freq='1min')})


    ts_cv = cv.TimeSeriesSplit(eval_periods, train_days=train_days, gap=gap, gap_unit=gap_unit)

    for fold, (train_idx, valid_idx) in enumerate(ts_cv.split(dates)):
        train_date = dates.loc[train_idx, 'date']
        valid_date = dates.loc[valid_idx, 'date']
        train_start_date, train_end_date = train_date.agg(('min', 'max'))
        valid_start_date, valid_end_date = valid_date.agg(('min', 'max'))

        check_date(train_start_date, train_periods[fold][0])
        check_date(train_end_date, train_periods[fold][1])
        check_date(valid_start_date, eval_periods[fold][0])
        check_date(valid_end_date, eval_periods[fold][1])
    


    
