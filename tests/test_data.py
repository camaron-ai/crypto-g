from src import data
import pandas as pd
import numpy as np


def test_date_to_timestamp(raw_data: pd.DataFrame):
    time = pd.to_datetime(raw_data['timestamp'], unit='s')
    assert np.allclose(data.date_to_timestamp(time), raw_data['timestamp']), \
           'timestamp do not match'


def test_fill_gaps_with_timestmap():
    # create a min freq date range 
    expected_date_range = pd.date_range('2021-01-01', '2021-12-31', freq='min')
    # to timestamp
    timestamp = data.date_to_timestamp(expected_date_range)
    # create a dataframe
    df = pd.DataFrame({'timestamp': timestamp})
    # drop random dates but the first and last timestamp 
    # this is how we determine the time range
    drop_indices = np.random.choice(np.arange(1, len(df)-1), size=len(df)//2, replace=False)
    missing_df = df.drop(drop_indices)
    # apply function
    reconstructed_df = data.fill_gaps_with_timestmap(missing_df)
    # turn timestamp back to dates
    actual_date_range = pd.to_datetime(reconstructed_df['timestamp'], unit='s')
    assert (actual_date_range == expected_date_range).all()