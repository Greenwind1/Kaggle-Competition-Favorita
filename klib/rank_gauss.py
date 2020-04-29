# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy
from scipy.special import erfinv


def rankgauss(df):
    df = df.replace(np.inf, np.nan)
    df = df.replace(-np.inf, np.nan)
    num_df = df

    # In case of NN, this would be effective. (ireko)
    # num_df = num_df.replace(np.nan, 0)

    # print('ranking...')
    num_df = num_df.rank(axis=0) / num_df.shape[0]

    # print('scaling...')
    num_df = 2 * num_df - 1

    # print('replace min/max values')
    num_df = num_df.replace(-1, -0.99999)
    num_df = num_df.replace(1, 0.99999)

    # print('erfinv')
    num_df = erfinv(num_df)

    return num_df


class GaussRankScaler(object):

    def __init__(self):
        self.epsilon = 1e-9
        self.lower = -1 + self.epsilon
        self.upper = 1 - self.epsilon
        self.range = self.upper - self.lower

    def fit_transform(self, X):
        i = np.argsort(X, axis=0)
        j = np.argsort(i, axis=0)

        assert (j.min() == 0).all()
        assert (j.max() == len(j) - 1).all()

        j_range = len(j) - 1
        divider = j_range / self.range

        transformed = j / divider
        transformed = transformed - self.upper
        transformed = scipy.special.erfinv(transformed)

        return transformed


if __name__ == '__init__':
    scaler = GaussRankScaler()
    for feat in tqdm(features):
        train[feat] = scaler.fit_transform(train[feat])
