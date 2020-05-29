# -*- coding: utf-8 -*-
"""

coded by HoxoMaxwell
 
"""
import os
import gc
import sys
import warnings
import psutil
import json
import pickle
import collections as cl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None
# dir(pd.options.display)
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')
sys.path.append('')


def train_generator(df, promo_df, items, stores,
                    timesteps, first_pred_start,
                    n_range=1, day_skip=7, batch_size=2000,
                    aux_as_tensor=False, reshape_output=0,
                    first_pred_start_2016=None):
    # ----------------------------------------------------------------
    # | X = timesteps days | pred_start (n_r * d_skip) | first_pred_start
    # | X = timesteps days | y = 16 days | ----------- | first_pred_start
    # ----------------------------------------------------------------
    encoder = LabelEncoder()
    items_reindex = items.reindex(df.index.get_level_values(1))
    item_family = encoder.fit_transform(items_reindex['family'].values)
    item_class = encoder.fit_transform(items_reindex['class'].values)
    item_perish = items_reindex['perishable'].values

    stores_reindex = stores.reindex(df.index.get_level_values(0))
    store_nbr = df.reset_index().store_nbr.values - 1
    store_cluster = stores_reindex['cluster'].values - 1
    store_type = encoder.fit_transform(stores_reindex['type'].values)

    item_group_mean = df.groupby('item_nbr').mean()
    store_group_mean = df.groupby('store_nbr').mean()

    cat_features = np.stack([item_family, item_class, item_perish,
                             store_nbr, store_cluster, store_type],
                            axis=1)

    while 1:
        # permutation of [0, 1, 2, 3, ..., 15]
        date_part = np.random.permutation(range(n_range))

        if first_pred_start_2016 is not None:
            range_diff = \
                (first_pred_start - first_pred_start_2016).days / day_skip
            date_part = np.concatenate([
                date_part,
                np.random.permutation(
                    range(range_diff, int(n_range / 2) + range_diff)
                )
            ])

        for i in date_part:
            keep_idx = np.random.permutation(df.shape[0])[:batch_size]
            df_tmp = df.iloc[keep_idx, :]
            promo_df_tmp = promo_df.iloc[keep_idx, :]
            cat_features_tmp = cat_features[keep_idx]

            pred_start = first_pred_start - timedelta(days=int(day_skip * i))

            # Generate a batch of random subset data.
            # All data in the same batch are in the same period.
            yield create_dataset_part(
                df_tmp, promo_df_tmp, cat_features_tmp,
                item_group_mean=item_group_mean,
                store_group_mean=store_group_mean,
                timesteps=timesteps,
                pred_start=pred_start,
                reshape_output=reshape_output,
                aux_as_tensor=aux_as_tensor,
                is_train=True
            )

            gc.collect()


def create_dataset_part(df, promo_df, cat_features,
                        item_group_mean, store_group_mean,
                        timesteps, pred_start,
                        reshape_output, aux_as_tensor, is_train,
                        weight=False):
    item_mean_df = item_group_mean.reindex(df.index.get_level_values(1))
    store_mean_df = store_group_mean.reindex(df.index.get_level_values(0))

    # sales features (None, 200)
    x, y = create_xy_span(df, pred_start, timesteps, is_train)
    is0 = (x == 0).astype('uint8')

    dataset_date_range = pd.date_range(
        start=pred_start - timedelta(days=timesteps),
        periods=timesteps + 16
    )
    # promo features (None, 216)
    promo = promo_df[dataset_date_range].values

    # weekday features (None, 216)
    weekday = np.tile(
        A=[d.weekday() for d in dataset_date_range],
        reps=(x.shape[0], 1)
    )

    # day of month features (None, 216)
    dom = np.tile(
        A=[d.day - 1 for d in dataset_date_range],
        reps=(x.shape[0], 1)
    )

    # item_mean features (None, 200)
    item_mean, _ = create_xy_span(item_mean_df, pred_start, timesteps, False)

    # store features (None, 200)
    store_mean, _ = create_xy_span(store_mean_df, pred_start, timesteps, False)

    # df_year_ago, _ = create_xy_span(df, pred_start - timedelta(days=365),
    #                                 timesteps + 16, False)

    # quarter_ago features (None, 216)
    df_quarter_ago, _ = create_xy_span(df, pred_start - timedelta(days=91),
                                       timesteps + 16, False)

    if reshape_output > 0:
        x = x.reshape(-1, timesteps, 1)
    if reshape_output > 1:
        is0 = is0.reshape(-1, timesteps, 1)
        promo = promo.reshape(-1, timesteps + 16, 1)
        # df_year_ago = df_year_ago.reshape(-1, timesteps + 16, 1)
        df_quarter_ago = df_quarter_ago.reshape(-1, timesteps + 16, 1)
        item_mean = item_mean.reshape(-1, timesteps, 1)
        store_mean = store_mean.reshape(-1, timesteps, 1)

    w = (cat_features[:, 2] * 0.25 + 1) / (cat_features[:, 2] * 0.25 + 1).mean()

    cat_features = np.tile(
        cat_features[:, None, :], (1, timesteps + 16, 1)
    ) if aux_as_tensor else cat_features

    # Use when only 6th-16th days (private periods) are in the training output
    # if is_train: y = y[:, 5:]

    if weight:
        return (
            [x, is0, promo,
             # df_year_ago,
             df_quarter_ago, weekday, dom,
             cat_features,
             item_mean, store_mean], y, w)
    else:
        return (
            [x, is0, promo,
             # df_year_ago,
             df_quarter_ago, weekday, dom,
             cat_features,
             item_mean, store_mean], y)


def create_xy_span(df, pred_start, timesteps, is_train=True):
    X = df[pd.date_range(
        pred_start - timedelta(days=timesteps),
        pred_start - timedelta(days=1)
    )].values

    if is_train:
        y = df[pd.date_range(pred_start, periods=16)].values
    else:
        y = None
    return X, y
