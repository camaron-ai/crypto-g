import numpy as np
from typing import Tuple
import pandas as pd


ASSET_WEIGHT = {
'Bitcoin Cash': 2.3978952727983707,
'Binance Coin': 4.30406509320417,
'Bitcoin': 6.779921907472252,
'EOS.IO': 1.3862943611198906,
'Ethereum Classic': 2.079441541679836,
'Ethereum': 5.8944028342648505,
'Litecoin': 2.3978952727983707,
'Monero': 1.6094379124341005,
'TRON': 1.791759469228055,
'Stellar': 2.079441541679836,
'Cardano': 4.406719247264253,
'IOTA': 1.0986122886681098,
'Maker': 1.0986122886681098,
'Dogecoin': 3.555348061489413}


TOTAL_WEIGHT_SUM = sum(ASSET_WEIGHT.values())

#### weighted correlation cofficient
def compute_weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return np.average(x, weights=w)


def compute_weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    mean = compute_weighted_mean(x, w)
    dev = np.square(x - mean)
    return compute_weighted_mean(dev, w)


def compute_weighted_cov(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    y_mean = compute_weighted_mean(y, w)
    yhat_mean = compute_weighted_mean(yhat, w)
    return compute_weighted_mean((y - y_mean) * (yhat - yhat_mean), w)


def compute_weighted_corr(y: np.ndarray, yhat: np.ndarray,
                          w: np.ndarray = None) -> float:
    if w is None:
        w = np.ones(len(y))
    assert len(y) == len(yhat)
    var_y = compute_weighted_var(y, w)
    var_yhat = compute_weighted_var(yhat, w)
    
    return compute_weighted_cov(y, yhat, w) / np.sqrt(var_y * var_yhat)


def compute_correlation(df: pd.DataFrame,
                        target_name: str = 'Target',
                        yhat_name: str = 'yhat',
                        group_col: str = 'Asset_ID') -> pd.DataFrame:
    def _spearman_corr(d: pd.DataFrame):
        return np.corrcoef(d[target_name], d[yhat_name])[0, 1]
    
    assert df[target_name].isna().sum() == 0, f'{target_name} includes NaN'
    corrs = df.groupby(group_col).apply(_spearman_corr)
    return corrs.to_frame('corr').reset_index()


def compute_sharpe(df: pd.DataFrame,
                   period: int = 60*24*7,   # weekly
                   target_name: str = 'Target',
                   yhat_name: str = 'yhat',
                   weight_name: str = 'weight',
                   ) -> float:
    
    timesteps = (df['timestamp'].max() - df['timestamp']) // 60   # from 0 up to n min,
    time_groups = timesteps // period
    corrs = df.groupby(time_groups).apply(lambda d: compute_weighted_corr(y=d[target_name].to_numpy(),
                                                                          yhat=d[yhat_name].to_numpy(),
                                                                          w=d[weight_name].to_numpy()))
    assert np.isnan(corrs).sum() == 0, 'period corrs contains NaN values'
    mean = corrs.mean()
    std = corrs.std()
    consistency = (corrs > 0.001).mean()
    return {'sharpe': mean / (std + 1e-15),
            'corr_period_mean': mean,
            'corr_period_std': std,
            'consistency': consistency,
            'min_period_corr': corrs.min()}


def compute_metrics(df: pd.DataFrame,
                    target_name: str = 'Target',
                    yhat_name: str = 'yhat',
                    group_col: str = 'Asset_Name') -> Tuple[pd.Series, pd.DataFrame]:

    # BASE APPROACH, COMPUTE CORR AND THE WEIGHTED THEM
    corrs_df = compute_correlation(df, target_name=target_name,
                                   yhat_name=yhat_name,
                                   group_col=group_col)    
    corr_stats = corrs_df['corr'].agg(('min', 'max', 'std')).add_prefix('corr_').to_dict()
    # COMPUTE WEIGHTED CORRELATION USING FORMULA
    df['_weight'] = df[group_col].map(ASSET_WEIGHT)
    theor_corr = compute_weighted_corr(y=df[target_name], yhat=df[yhat_name], w=df['_weight'].to_numpy())
    # DIVIDE IT INTO DAILY CHUNKS AND COMPUTE SHARPE
    sharpe_scores = compute_sharpe(df, target_name=target_name, yhat_name=yhat_name, weight_name='_weight')
    scores = {'corr': theor_corr,
              'crypto_consistency': (corrs_df['corr'] >= 0.001).sum()}
    scores.update(sharpe_scores)
    scores.update(corr_stats)
    df.drop('_weight', axis=1, inplace=True)
    return pd.Series(scores)