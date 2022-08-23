# from src.preprocessing import feature_gen, ingest_data
# import pytest
# import pandas as pd
# import numpy as np


# @pytest.fixture
# def bitcoin_data(pytestconfig, raw_data: pd.DataFrame):
#     asset = 1
#     data = raw_data.query("Asset_ID==@asset").reset_index(drop=True)
#     data = ingest_data.fill_gaps_crypto_data(data)
#     data = ingest_data.infer_dtypes(data)
#     data = feature_gen.compute_base_features(data)
#     if pytestconfig.getoption("sample_size") == 'small':
#         data = data.tail(24*60).reset_index(drop=True) # day of data
#     return data



# @pytest.mark.parametrize('window', [60])
# def test_compute_features_equivalent(bitcoin_data: pd.DataFrame, window: int):
#     # using rolling pandas
#     expected = feature_gen.compute_features_on_train(bitcoin_data, window, feature_gen.FEATURE_DICT)
#     # manually computing with fixed intervals
#     actual = pd.concat([feature_gen.compute_features_on_inference(bitcoin_data.iloc[:i+1], n=window,
#                                                                   feature_dict=feature_gen.FEATURE_DICT)
#                         for i in range(len(bitcoin_data))])

#     assert np.isin(actual.columns, expected.columns).all()
#     assert np.allclose(expected, actual[expected.columns])
