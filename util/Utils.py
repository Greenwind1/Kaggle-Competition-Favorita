import gc
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from datetime import date, timedelta


def load_data():
    # ---------------------------------------------------------------
    #   Raw train data
    # ---------------------------------------------------------------
    # df_train = pd.read_csv(
    #     './input/train.csv',
    #     usecols=[1, 2, 3, 4, 5],
    #     # dtype={'onpromotion': bool},
    #     converters={
    #         'unit_sales': lambda u: np.log1p(float(u)) if float(u) > 0 else 0
    #     },
    #     parse_dates=["date"]
    # )

    # df_train.to_feather('./input/train.f')
    # df_train = pd.read_feather('./input/train.f', use_threads=True)

    # ---------------------------------------------------------------
    #   Extract data from 2016-01-01
    #   2013-01-01 <= train.date <= 2017-08-15, 1684 days
    # ---------------------------------------------------------------
    # df_train = df_train.loc[df_train.date >= pd.datetime(2016, 1, 1)]
    # df_train.reset_index().to_feather('./input/train_2016.f')

    df_train = pd.read_feather(
        './input/train_2016.f',
        columns=['date', 'store_nbr', 'item_nbr', 'unit_sales', 'onpromotion'],
        use_threads=True
    )

    df_test = pd.read_csv(
        './input/test.csv',
        usecols=[0, 1, 2, 3, 4],
        dtype={'onpromotion': bool},
        parse_dates=["date"]
    ).set_index(['store_nbr', 'item_nbr', 'date'])

    gc.collect()

    # promo
    promo_2017_train = df_train.set_index(
        ["store_nbr", "item_nbr", "date"]
    )[["onpromotion"]].unstack(level=-1).fillna(False)
    promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)

    promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
    promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
    promo_2017_test = \
        promo_2017_test.reindex(promo_2017_train.index).fillna(False)

    promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
    del promo_2017_test, promo_2017_train
    gc.collect()

    # melt (from long to wide)
    df_train = df_train.set_index(
        ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
    df_train.columns = df_train.columns.get_level_values(1)

    # items
    # items = pd.read_csv('./input/items.csv').set_index("item_nbr")
    # stores = pd.read_csv('./input/stores.csv').set_index("store_nbr")
    # items = items.reindex(df_2017.index.get_level_values(1))

    return df_train, promo_2017


def save_unstack(df, promo, df_name, promo_name):
    df.columns = df.columns.astype('str')
    df.reset_index().to_feather(df_name)
    promo.columns = promo.columns.astype('str')
    promo.reset_index().to_feather(promo_name)


def load_unstack(df_name, promo_name):
    df_2017 = pd.read_feather(df_name).set_index(['store_nbr', 'item_nbr'])
    df_2017.columns = pd.to_datetime(df_2017.columns)
    promo_2017 = pd.read_feather(promo_name).set_index(
        ['store_nbr', 'item_nbr']
    )
    promo_2017.columns = pd.to_datetime(promo_2017.columns)
    items = pd.read_csv('./input/items.csv').set_index('item_nbr')
    stores = pd.read_csv('./input/stores.csv').set_index('store_nbr')

    return df_2017, promo_2017, items, stores


def train_generator(df, promo_df, items, stores,
                    timesteps, first_pred_start,
                    n_range=16, day_skip=7, batch_size=2000,
                    aux_as_tensor=False, reshape_output=0,
                    first_pred_start_2016=None,
                    weight=False):
    # ----------------------------------------------------------------
    # | X = timesteps days | pred_start (n_r * d_skip) | first_pred_start
    # | X = timesteps days | y = 16 days | ----------- | first_pred_start
    #
    #   item kinds = 3841
    #   1 batch: 2,000 instances x 16 cycles = 32,000 instances for train
    #   1 epoch: 160,964 / 2,000 = 81 batches
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

    cat_features = np.stack([
        item_family, item_class, item_perish,
        store_nbr, store_cluster, store_type
    ], axis=1)

    while 1:
        # permutation of [0, 1, 2, 3, ..., 15]
        cycle = np.random.permutation(range(n_range))

        if first_pred_start_2016 is not None:
            range_diff = \
                (first_pred_start - first_pred_start_2016).days / day_skip
            cycle = np.concatenate([
                cycle,
                np.random.permutation(
                    range(range_diff, int(n_range / 2) + range_diff)
                )
            ])

        # usually step to this
        # this loop makes 2,000 x 16 cycles = 32,000 instances for train
        for c in cycle:
            keep_idx = np.random.permutation(df.shape[0])[:batch_size]
            df_tmp = df.iloc[keep_idx, :]
            promo_df_tmp = promo_df.iloc[keep_idx, :]
            cat_features_tmp = cat_features[keep_idx, :]

            pred_start = first_pred_start - timedelta(days=int(day_skip * c))

            # Generate a batch of random subset data.
            # All data in the same batch are in the same period.
            yield create_dataset_part(
                df_tmp=df_tmp,
                promo_df_tmp=promo_df_tmp,
                cat_features_tmp=cat_features_tmp,
                item_group_mean=item_group_mean,
                store_group_mean=store_group_mean,
                timesteps=timesteps,
                pred_start=pred_start,
                reshape_output=reshape_output,
                aux_as_tensor=aux_as_tensor,
                is_train=True,
                weight=weight,
            )

            gc.collect()


def create_dataset_part(df_tmp, promo_df_tmp, cat_features_tmp,
                        item_group_mean, store_group_mean,
                        timesteps, pred_start,
                        reshape_output,
                        aux_as_tensor,
                        is_train,
                        weight=False):
    item_mean_df = item_group_mean.reindex(df_tmp.index.get_level_values(1))
    store_mean_df = store_group_mean.reindex(df_tmp.index.get_level_values(0))

    # sales features (None, 200)
    x, y = create_xy_span(df_tmp, pred_start, timesteps, is_train)
    # 0 sales features (None, 200)
    is0 = (x == 0).astype('uint8')

    # date range for train + test
    date_range_all = pd.date_range(
        start=pred_start - timedelta(days=timesteps),
        periods=timesteps + 16  # 216
    )

    # promo features (None, 216)
    promo = promo_df_tmp[date_range_all].values

    # weekday features (None, 216), NOT used
    weekday = np.tile(
        A=[d.weekday() for d in date_range_all],
        reps=(x.shape[0], 1)
    )

    # day of month features (None, 216), NOT used
    dom = np.tile(
        A=[d.day - 1 for d in date_range_all],
        reps=(x.shape[0], 1)
    )

    # item_mean features (None, 200)
    item_mean, _ = create_xy_span(item_mean_df, pred_start, timesteps, False)

    # store features (None, 200)
    store_mean, _ = create_xy_span(store_mean_df, pred_start, timesteps, False)

    # df_year_ago, _ = create_xy_span(df, pred_start - timedelta(days=365),
    #                                 timesteps + 16, False)

    # quarter_ago features (None, 216)
    df_quarter_ago, _ = create_xy_span(df_tmp, pred_start - timedelta(days=91),
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

    w = (cat_features_tmp[:, 2] * 0.25 + 1) / (
                cat_features_tmp[:, 2] * 0.25 + 1).mean()

    # transform cat features into seq if aux_as_tensor is True
    cat_features_tmp = np.tile(
        cat_features_tmp[:, np.newaxis, :], (1, timesteps + 16, 1)
    ) if aux_as_tensor else cat_features_tmp

    if weight:
        return ([
                    x,
                    is0,
                    promo,
                    # df_year_ago,
                    df_quarter_ago,
                    weekday,
                    dom,
                    cat_features_tmp,
                    item_mean,
                    store_mean,
                ], y, w)
    else:
        return ([
                    x,  # shape: (2000, 200, 1)
                    is0,  # shape: (2000, 200, 1)
                    promo,  # shape: (2000, 216, 1)
                    # df_year_ago,
                    df_quarter_ago,  # shape: (2000, 216, 1)
                    weekday,  # shape: (2000, 216, 1)
                    dom,  # shape: (2000, 216)
                    cat_features_tmp,  # shape: (2000, 6)
                    item_mean,  # shape: (2000, 200, 1)
                    store_mean  # shape: (2000, 200, 1)
                ], y)
        # y shape: (2000, 16)


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


# Create validation and test data
def create_dataset(df, promo_df, items, stores,
                   timesteps, first_pred_start,
                   is_train=True, aux_as_tensor=False, reshape_output=0):
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

    cat_features = np.stack(
        [item_family, item_class, item_perish, store_nbr, store_cluster,
         store_type], axis=1)

    return create_dataset_part(df, promo_df, cat_features, item_group_mean,
                               store_group_mean, timesteps, first_pred_start,
                               reshape_output, aux_as_tensor, is_train)


# Not used in the final model
def random_shift_slice(mat, start_col, timesteps, shift_range):
    shift = np.random.randint(shift_range + 1, size=(mat.shape[0], 1))
    shift_window = np.tile(shift, (1, timesteps)) + np.tile(
        np.arange(start_col, start_col + timesteps), (mat.shape[0], 1))
    rows = np.arange(mat.shape[0])
    rows = rows[:, None]
    columns = shift_window
    return mat[rows, columns]


# Calculate RMSE scores for all 16 days,
# first 5 days (for public LB) and 6th-16th days (for private LB)
def cal_score(actual, pred):
    return [
        np.round(metrics.mean_squared_error(actual, pred), 4),
        np.round(metrics.mean_squared_error(actual[:, :5], pred[:, :5]), 4),
        np.round(metrics.mean_squared_error(actual[:, 5:], pred[:, 5:]), 4)
    ]


# Create submission file
def make_submission(df_index, test_pred, filename):
    df_test = pd.read_csv(
        './input/test.csv',
        usecols=[0, 1, 2, 3, 4],
        dtype={'onpromotion': bool},
        parse_dates=["date"]
    ).set_index(['store_nbr', 'item_nbr', 'date'])
    df_preds = pd.DataFrame(
        test_pred,
        index=df_index,
        columns=pd.date_range("2017-08-16", periods=16)
    ).stack().to_frame("unit_sales")
    df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

    submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
    submission["unit_sales"] = \
        np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
    submission.to_csv(filename, float_format='%.5f', index=None)
