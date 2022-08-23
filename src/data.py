import pandas as pd
import numpy as np
import os

# INGEST DATA
# DATASET DTYPES FOR SAVING MEMORY
DTYPES = {'Asset_ID': 'int32',
          'Open': 'float32',
          'High': 'float32',
          'Low': 'float32',
          'Close': 'float32',
          'VWAP': 'float32'}


def merge_asset_details(df: pd.DataFrame, asset_details_path: str) -> pd.DataFrame:
    asset_details = pd.read_csv(asset_details_path)
    df = df.merge(asset_details[['Asset_ID', 'Asset_Name']], on='Asset_ID', how='left')
    assert df['Asset_Name'].isna().sum() == 0, 'unexpected Asset ID'
    return df


def infer_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # replace inf with NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.astype(DTYPES)


def date_to_timestamp(dates: pd.Series) -> pd.Series:
    return dates.astype(np.int64) // 10 ** 9


def create_valid_timestamp_range(data: pd.DataFrame, dt_col: str = 'timestamp') -> np.ndarray:
    start_ttp, end_ttp = data[dt_col].agg(('min', 'max'))
    return np.arange(start_ttp, end_ttp+60, 60)
    

def fill_gaps_with_timestmap(data: pd.DataFrame, dt_col: str = 'timestamp') -> pd.DataFrame:
    assert data[dt_col].duplicated().sum() == 0, f'{dt_col} contains duplicates, cant reindex from duplicated values'
    valid_ttp_range = create_valid_timestamp_range(data, dt_col)
    data = data.set_index(dt_col)
    filled_data = data.reindex(valid_ttp_range)
    return filled_data.reset_index().rename(columns={'index': dt_col})


def fill_gaps_crypto_data(data: pd.DataFrame,
                          dt_col: str = 'timestamp') -> pd.DataFrame:
    
    asset_id = np.unique(data['Asset_ID'])
    assert len(asset_id) == 1, 'expected one Asset_ID'
    data = fill_gaps_with_timestmap(data, dt_col)
    data['Asset_ID'] = int(asset_id[0])
    return data


def get_mask_for_asset(data: pd.DataFrame, asset_id: int) -> pd.Series:
    return (data['Asset_ID'] == asset_id)


def get_data_for_asset(data: pd.DataFrame, asset_id: int) -> pd.DataFrame:
    mask = get_mask_for_asset(data, asset_id)
    return data.loc[mask, :].reset_index(drop=True)


def ingest_raw_data(data: pd.DataFrame, asset_details_path: str) -> pd.DataFrame:
    assert os.path.exists(asset_details_path)
    data = infer_dtypes(data)
    data = data.dropna(subset=['Target'])
    data['time'] = pd.to_datetime(data['timestamp'], unit='s')
    data = merge_asset_details(data, asset_details_path)
    data.sort_values(by=['Asset_ID', 'time'], inplace=True)
    return data.reset_index(drop=True)
