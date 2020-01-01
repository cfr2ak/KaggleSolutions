import torch
import pandas as pd
import numpy as np


def preprocess(train_raw, test_raw):
    train = []
    test = []
    for index, row in train_raw.iterrows():
        ndarr = row[1:].values
        ndarr = ndarr / 255.
        train.append(torch.from_numpy(ndarr).float())
    for index, row in test_raw.iterrows():
        ndarr = row.values
        ndarr = ndarr / 255.
        test.append(torch.from_numpy(ndarr).float())

    return train, test


def load_data():
    train_raw = pd.read_csv('../../data/train.csv')
    test_raw = pd.read_csv('../../data/test.csv')

    train, test = preprocess(train_raw, test_raw)

    return train, test


