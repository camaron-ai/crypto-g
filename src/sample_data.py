import pandas as pd
import click
from pathlib import Path

def filter_by_date(data: pd.DataFrame,
                   start_date: str,
                   time_col: str = 'timestamp'):
    index = data[time_col] >= start_date
    return data.loc[index, :]


TEST_SET_END_DATE = '2021-09-13'
SAMPLE_LEVELS = ['2021-01-01', '2021-09-09'] 
def create_samples(data_dir: str, output_dir: str):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    train_file = data_dir / 'train.csv'
    assert train_file.exists(), f'the train file do not exist in {data_dir}'
    print('loading train set')
    data = pd.read_csv(train_file)
    print(f'shape: {data.shape}')
    data['date'] = pd.to_datetime(data['timestamp'], unit='s')
    last_date = data['date'].max()

    # ends at TEST_SET_END_DATE
    data = data.loc[data['date'] <= TEST_SET_END_DATE, :].reset_index(drop=True)

    # toy directory

    for level, start_date in enumerate(SAMPLE_LEVELS, start=1):
        print(level, start_date)
        sample_dir = output_dir.joinpath('sample', str(level))
        sample_dir.mkdir(exist_ok=True, parents=True)
        # last 21 day worth of data
        print('filtering sample')
        sample_data = filter_by_date(data, start_date, time_col='date')
        print(f'{sample_data.date.min()} -> {sample_data.date.max()}')
        sample_data = sample_data.drop('date', axis=1)
        print(f'shape: {sample_data.shape}')
        print(f'saving sample on {sample_dir}')
        sample_data.to_csv(sample_dir / 'train.csv', index=False)


if __name__ == '__main__':
    create_samples('data/raw/', 'data/raw/')

    