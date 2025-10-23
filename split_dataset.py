"""Utility script to split dataset/PhiUSIIL_Phishing_URL_Dataset.csv into train/test (70/30).
Usage: python split_dataset.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def main():
    data_path = Path('dataset') / 'PhiUSIIL_Phishing_URL_Dataset.csv'
    train_out = Path('dataset') / 'train.csv'
    test_out = Path('dataset') / 'test.csv'

    if not data_path.exists():
        raise SystemExit(f"Dataset not found at {data_path.resolve()}")

    df = pd.read_csv(data_path)
    print(f'Loaded dataset with {len(df)} rows and {len(df.columns)} columns')

    train, test = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)
    print(f'Train rows: {len(train)}, Test rows: {len(test)}')

    train.to_csv(train_out, index=False)
    test.to_csv(test_out, index=False)
    print(f'Wrote {train_out} and {test_out}')


if __name__ == '__main__':
    main()
