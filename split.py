import os

import pandas as pd
from sklearn.model_selection import train_test_split

dataset_path = 'resources/'


def read_dataset():
    if not os.path.exists(dataset_path):
        raise RuntimeError('The similarity dataset has not found')
    all_path = os.path.join(dataset_path, 'all.csv')
    return pd.read_csv(all_path, sep='|', encoding='utf-8-sig')


def split_train_test(dataset):
    train_samples, test_samples = train_test_split(dataset, train_size=0.75, test_size=0.25, random_state=42,
                                                   shuffle=True)
    return train_samples, test_samples


def save(dataset, filename):
    dataset = pd.DataFrame(dataset)
    filepath = os.path.join(dataset_path, filename)
    dataset.to_csv(filepath, sep='|', encoding='utf-8-sig', index_label='index')


if __name__ == '__main__':
    samples = read_dataset()
    train, test = split_train_test(samples)
    save(train, 'train.csv')
    save(test, 'dev.csv')
