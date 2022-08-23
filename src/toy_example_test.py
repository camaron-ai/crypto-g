import pandas as pd
from pathlib import Path


if __name__ == '__main__':
    DATA_DIR = Path('data/raw/')
    TOY_SAMPLE = DATA_DIR.joinpath('toy_sample', 'train.csv')

    df = pd.read_csv(TOY_SAMPLE)
    df['row_id'] = list(range(len(df)))
    group_num = df[['timestamp']].drop_duplicates().sort_values(by='timestamp')
    group_num['group_num'] = list(range(len(group_num)))
    df = df.merge(group_num, on='timestamp', how='left')
    assert df['group_num'].isna().sum() == 0
    df.to_csv(DATA_DIR / 'toy_example_test.csv', index=False)
