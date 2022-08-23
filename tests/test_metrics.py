from src import metrics
import numpy as np
import pytest


SIZES = [1000, 500, 5, 320]
@pytest.mark.parametrize('size', SIZES)
def test_weighted_corr(size: int):
    y = np.random.randn(size)
    noise = np.random.randn(size) * 0.01
    coeff, bias = np.random.randn(2)
    yhat = coeff * y + bias + noise
    w = np.ones(size)
    actual_corr = metrics.compute_weighted_corr(y, yhat)
    expected_corr = np.corrcoef(y, yhat)[0, 1]

    assert np.allclose(expected_corr, actual_corr), \
    f'corr do not match, diff {(expected_corr-actual_corr):.3f}'
