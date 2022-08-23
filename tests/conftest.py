import pytest
import pandas as pd
from pathlib import Path

RAW_TRAIN_DIR = Path('data/raw/')


def pytest_addoption(parser):
    parser.addoption(
        "--sample_level",
        action="store",
        default=1,
        choices=[0, 1, 2],
        type=int,
        help="what sample size to use"
)


@pytest.fixture
def raw_data(pytestconfig):
    sample_level = pytestconfig.getoption("sample_level")

    path = RAW_TRAIN_DIR
    if sample_level > 0:
        path = path.joinpath('sample', str(sample_level))
    return pd.read_csv(path / 'train.csv')
