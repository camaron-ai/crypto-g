import pandas as pd
import time
import numpy as np
import gc
from typing import Tuple, Dict, Any
# from src.pipeline.feature_gen import compute_instant_features #NO_IMPORT
from src.data import infer_dtypes #NO_IMPORT


EXPECTED_RAW_COLS = ['timestamp', 'Asset_ID', 'Count',
                     'Open', 'High', 'Low', 'Close',
                     'Volume', 'VWAP']

# def process_train_data(df: pd.DataFrame,
#                        window: int = 60) -> pd.DataFrame:
#     asset_ids = sorted(df['Asset_ID'].unique())
    
#     global_features = []
#     for asset_id in asset_ids:
#         print(f'processing asset_id={asset_id}')
#         raw_local_data = df.query("Asset_ID==@asset_id").reset_index(drop=True)
#         # fill nan gaps
#         raw_local_data = fill_gaps_crypto_data(raw_local_data)
#         raw_local_data = infer_dtypes(raw_local_data)
#         # base features
#         raw_features = compute_base_features(raw_local_data)
        
#         # compute history features
#         start_time = time.time()
#         features = compute_features_on_train(raw_features, window, FEATURE_DICT)
#         elapsed_time = (time.time() - start_time) / 60
        
#         print(f'elapsed time: {elapsed_time:.4f}min')
#         # add timestamp
#         features['timestamp'] = raw_features['timestamp'].to_numpy()
#         features['Asset_ID'] = asset_id
#         global_features.append(features)

#         del raw_local_data, raw_features
#         gc.collect()
#     print('joining datasets')
#     global_features = pd.concat(global_features, axis=0, ignore_index=True)
#     assert global_features['Asset_ID'].nunique() == len(asset_ids), \
#            f'missing Asset_IDs'
#     return global_features


# def process_test_data(test_dict: Dict[str, float], local_history_df: pd.DataFrame,
#                       window: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame]:

#     last_timestamp = local_history_df.iloc[-1]['timestamp']
#     current_timestamp = test_dict['timestamp']
#     # add new observation forget the last first row
#     local_history_df = local_history_df.append([test_dict], ignore_index=True)
#     minute_diff = (current_timestamp - last_timestamp) // 60

#     assert minute_diff > 0, f'current timestamp included in history df, {current_timestamp} <= {last_timestamp}'

#     if minute_diff > 1:
#         print(f'missing more than one minut of data, missing minutes: {minute_diff}')
#         print(f'filling gaps')
#         local_history_df = fill_gaps_crypto_data(local_history_df)
#     raw_features = compute_base_features(local_history_df)
#     features = compute_features_on_inference(raw_features, n=window, feature_dict=FEATURE_DICT)

#     return features, local_history_df


def test_submission_format(submission: pd.DataFrame, expected_len: int = 14):
    assert list(submission.columns) == ['row_id', "Target"], 'submission do not match expected columns'
    assert len(submission) == expected_len, 'submission do not match expected lenght'
    assert submission['Target'].isna().sum() == 0, 'target includes NaNs'
    assert submission['row_id'].dtype == np.int32
    assert submission['Target'].dtype == np.float64
    assert submission['Target'].isna().sum() == 0, 'submission contains NaN values'
    assert np.isinf(submission['Target']).sum() == 0 ,'submission contains inf values'


def inference(test_data: pd.DataFrame, submission: pd.DataFrame,
             models: Dict[str, Any],
             ) -> pd.DataFrame:
    expected_len = len(submission)
    test_data = infer_dtypes(test_data)
    features = compute_instant_features(test_data.loc[:, EXPECTED_RAW_COLS])
    records = features.to_dict('records')
    for index, asset_features in enumerate(records):
        # get the asset ID
        asset_id = int(asset_features['Asset_ID'])
        assert asset_id in models, f'{asset_id} not in TRAINED MODELS'
        # get model
        model = models[asset_id]
        asset_frame = pd.DataFrame([asset_features])
        local_test_yhat = model.predict(asset_frame)
        # add to submission format
        submission.iloc[index, 1] = local_test_yhat[0]
    # testing submission format
    test_submission_format(submission, expected_len=expected_len)
    return submission