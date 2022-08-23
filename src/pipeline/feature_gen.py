import pandas as pd
import numpy as np
from typing import List, Callable, Dict
from scipy.stats import linregress


feature_dict_dtype = Dict[str, List[Callable]]
RAW_FEATURES = ['Count', 'Open', 'High', 'Low', 'Close',
                'Volume', 'VWAP']
BASE_FEATURES_TO_DROP = ['Open']
RAW_BITCOIN_FEATURES = ['Close', 'Volume', 'Count']
BITCOIN_FEATURES = ['bitcoin__' + f.lower() for f in RAW_BITCOIN_FEATURES]
BITCOIN_FEATURES_MAPPER = dict(zip(RAW_BITCOIN_FEATURES, BITCOIN_FEATURES))

Asset_ID_WEIGHT_MAPPER = {2: 0.05865714877447485,
 0: 0.10528574344803326,
 1: 0.16584998207274143,
 5: 0.033911437045592165,
 7: 0.05086715556838825,
 6: 0.1441884755803646,
 9: 0.05865714877447485,
 11: 0.03936995920708227,
 13: 0.04382989655421098,
 12: 0.05086715556838825,
 3: 0.10779686228434221,
 8: 0.026874178031414907,
 10: 0.026874178031414907,
 4: 0.08697067905907711}


def assign_inplace(data: pd.DataFrame, values: Dict[str, float]) -> pd.DataFrame:
    for name, value in values.items():
        data.loc[:, name] = value
    return data


# FEATURE GEN FUNCTION
def log_return(x: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(x).diff(periods=periods).fillna(0)


def realized_volatility(series: pd.Series) -> float:
    return np.sqrt(np.sum(np.power(series.to_numpy(), 2)))


def linear_slope(series: pd.Series) -> float:
    linreg = linregress(np.arange(len(series)), series)
    return linreg.slope

# UTIL
def join_columns(columns):
    return list(map(lambda f: '__'.join(map(str, f)), columns))


def add_bitcoin_minute_features(df: pd.DataFrame, inference: bool = True):
    bitcoin_mask = (df['Asset_ID'] == 1)
    bitcoin_data = df.loc[bitcoin_mask, :].reset_index(drop=True)
    bitcoin_data = bitcoin_data.rename(columns=BITCOIN_FEATURES_MAPPER)
    if inference:
        assert len(df) == 14
        assert len(bitcoin_data) == 1
        return assign_inplace(df, bitcoin_data.loc[:, BITCOIN_FEATURES].iloc[0].to_dict())
    else:
        bitcoin_data = bitcoin_data.loc[:, ['timestamp'] + BITCOIN_FEATURES]
        return df.merge(bitcoin_data, on=['timestamp'], how='left')


def compute_market_minute_features(df: pd.DataFrame, inference: bool = False) -> pd.DataFrame:
    market_feat = df.loc[:, ['Asset_ID', 'timestamp']]
    market_feat['_weight'] = market_feat['Asset_ID'].map(Asset_ID_WEIGHT_MAPPER)
    market_feat['market_close_price'] = (df['Close'] * market_feat['_weight'])
    market_feat['market_volume'] = (df['Volume'] * market_feat['_weight'])
    market_feat['market_count'] = (df['Count'] * market_feat['_weight'])

    market_feat.drop('_weight', axis=1, inplace=True)
    if inference:
        assert len(market_feat) == 14
        market_feat = market_feat.drop(['Asset_ID', 'timestamp'], axis=1)
        agg_market_feat = market_feat.sum().to_dict()
        return assign_inplace(df, agg_market_feat)

    else:
        agg_market_feat = (market_feat.groupby(['Asset_ID', 'timestamp'])
                           .sum().reset_index())
        df = df.merge(agg_market_feat, on=['Asset_ID', 'timestamp'], how='left')

        return df


def compute_minute_features(df: pd.DataFrame, inference: bool = True) -> pd.DataFrame:
    assert np.isin(RAW_FEATURES, df.columns).all(), \
           'missing raw features'

    df['Volume'] = np.log1p(df['Volume'])
    df['Count'] =  np.sqrt(df['Count'])
    # create price features 
    df['upper_shadow'] = df['High'] / np.maximum(df['Close'], df['Open'])
    df['lower_shadow'] = np.minimum(df['Close'], df['Open']) / df['Low']

    # normalize High and Low features
    df['High'] = df['High'] / df['Open']
    df['Low'] = df['Low'] / df['Open']
    df['high_low_return'] = np.log1p(df['High'] / df['Low'])
    df['open_close_return'] = np.log1p(df['Close'] / df['Open'])

    # vol and count features
    df['dolar_amount'] = df['Close'] * df['Volume']
    df['vol_per_trades'] = df['Volume'] / df['Count']

    df = compute_market_minute_features(df, inference=inference)
    df = add_bitcoin_minute_features(df, inference=inference)
    return df.drop(BASE_FEATURES_TO_DROP, axis=1)



# # FEATURES TO COMPUTE
# FEATURE_DICT = {'High': [np.max],
#                 'Low': [np.min],
#                 'Close': [np.mean],
#                 'price_return_1': [np.sum, realized_volatility],
#                 'vwap_return_1': [np.sum, realized_volatility],
#                 'Count': [np.sum, np.max],
#                 'Volume': [np.sum, np.max],
#                 'high_low_return': [np.mean],
#                 'open_close_return': [np.mean],
#                }

# FEATURE_DICT = {
#                 'Close': [np.mean],
#                }



# def map_function_to_dataframe(X: pd.DataFrame,
#                  feature_dict: feature_dict_dtype) -> Dict[str, float]:
#     features = {f'{name}__{func.__name__}': func(X[name])
#                 for name, func_list in feature_dict.items()
#                 for func in func_list}
#     return features


# def compute_features_on_inference(X: pd.DataFrame, n: int,
#                                  feature_dict: feature_dict_dtype) -> pd.DataFrame:
#     features = map_function_to_dataframe(X.tail(n), feature_dict)
#     return pd.DataFrame([features]).add_suffix(SUFFIX_FOMRAT.format(n=n)).astype(np.float32)


# def compute_features_on_train(X: pd.DataFrame, n: int,
#                              feature_dict: feature_dict_dtype) -> pd.DataFrame:
#     assert X['Asset_ID'].nunique() == 1, \
#            'expected only one Asset_ID'
    
#     mov_features = X.rolling(n, min_periods=1).agg(feature_dict)
#     mov_features.columns = join_columns(mov_features.columns)
#     mov_features = mov_features.add_suffix(SUFFIX_FOMRAT.format(n=n))
    
#     assert len(mov_features) == len(X), 'output lenght do not match the input lenght'
#     return mov_features.astype(np.float32)