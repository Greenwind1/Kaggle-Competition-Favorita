import sys, psutil, os, warnings, gc, pickle

sys.path.append('..')

if len(sys.argv) < 2:
    sys.exit('Invalid args')
sys.path.append(sys.argv[1])

from datetime import date, timedelta
import pandas as pd
import numpy as np
import lightgbm as lgb
import psutil
import gc

from util.Utils import *

# -------------------------------------------------------------------
#   Env Setting
# -------------------------------------------------------------------
NAME = 'lgb_03'
SEED = int(sys.argv[2]) + 2020
# SEED = 2020 + 1
CPU = psutil.cpu_count()

# -------------------------------------------------------------------
#   Load Dataset
# -------------------------------------------------------------------
df_name = './input/unstack_train.f'
promo_name = './input/unstack_promo.f'
df, promo_df, items, stores = load_unstack(df_name, promo_name)
print('df shape =', df.shape, '\npromo_df shape =', promo_df.shape)
print('data span:', df.columns[0].strftime('%Y-%m-%d'),
      '-', df.columns[-1].strftime('%Y-%m-%d'), '\n')

df_data_range = pd.date_range(date(2017, 1, 1), date(2017, 8, 15))
promo_df = promo_df[df[df_data_range].max(axis=1) > 0]
promo_df = promo_df.astype('int')
df = df[df[df_data_range].max(axis=1) > 0]
print('df shape =', df.shape, '\npromo_df shape =', promo_df.shape)
print('data span:', df.columns[0].strftime('%Y-%m-%d'),
      '-', df.columns[-1].strftime('%Y-%m-%d'), '\n')

df_test = pd.read_csv(
    './input/test.csv',
    usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]
).set_index(['store_nbr', 'item_nbr', 'date'])
item_nbr_test = df_test.index.get_level_values(1)
item_nbr_train = df.index.get_level_values(1)
item_inter = list(set(item_nbr_train).intersection(set(item_nbr_test)))

df = df.loc[df.index.get_level_values(1).isin(item_inter)]
promo_df = promo_df.loc[promo_df.index.get_level_values(1).isin(item_inter)]
print('df shape =', df.shape, '\npromo_df shape =', promo_df.shape)
print('data span:', df.columns[0].strftime('%Y-%m-%d'),
      '-', df.columns[-1].strftime('%Y-%m-%d'))


# -------------------------------------------------------------------
#   Function
# -------------------------------------------------------------------
def get_timespan(df, dt, minus, periods, freq='D'):
    data_date_range = \
        pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)
    return df[data_date_range]


def prepare_dataset(train_end_point, is_train=True, one_hot=False):
    # shift, rolling, count0, promotion
    x = pd.DataFrame({
        # shift
        "1-shift": get_timespan(
            df, train_end_point, minus=1, periods=1
        ).values.ravel(),

        "2_shift": get_timespan(
            df, train_end_point, minus=2, periods=1
        ).values.ravel(),

        "3_shift": get_timespan(
            df, train_end_point, minus=3, periods=1
        ).values.ravel(),

        # rolling
        '3-shift_3-rolling_sum': get_timespan(
            promo_df, train_end_point, minus=3, periods=3
        ).sum(axis=1).values,

        "365-shift_16-rolling_mean": get_timespan(
            df, train_end_point, minus=365, periods=16
        ).mean(axis=1).values,

        "365-shift_16-rolling_count0": (get_timespan(
            df, train_end_point, minus=365, periods=16
        ) == 0).sum(axis=1).values,

        "365-shift_16-rolling_promo_sum": get_timespan(
            promo_df, train_end_point, minus=365, periods=16
        ).sum(axis=1).values
    })

    for i in [7, 14, 21, 30, 60, 90, 140, 365]:
        # shift + rolling
        x[f'{i}-shift_{i}-rolling_mean'] = get_timespan(
            df, train_end_point, minus=i, periods=i
        ).mean(axis=1).values

        x[f'{i}-shift_{i}-rolling_median'] = get_timespan(
            df, train_end_point, minus=i, periods=i
        ).median(axis=1).values

        x[f'{i}-shift_{i}-rolling_max'] = get_timespan(
            df, train_end_point, i, i
        ).max(axis=1).values

        x['mean_{}_haspromo'.format(i)] = \
            get_timespan(df, train_end_point, i, i)[
                get_timespan(promo_df, train_end_point, i, i) == 1
                ].mean(axis=1).values

        x['mean_{}_nopromo'.format(i)] = \
            get_timespan(df, train_end_point, i, i)[
                get_timespan(promo_df, train_end_point, i, i) == 0
                ].mean(axis=1).values

        x['count0_{}'.format(i)] = (get_timespan(
            df, train_end_point, i, i
        ) == 0).sum(axis=1).values

        x['promo_{}'.format(i)] = get_timespan(
            promo_df, train_end_point, i, i
        ).sum(axis=1).values

        item_mean = get_timespan(
            df, train_end_point, i, i
        ).mean(axis=1).groupby('item_nbr').mean().to_frame('item_mean')

        x['item_{}_mean'.format(i)] = df.join(item_mean)['item_mean'].values

        item_count0 = (get_timespan(
            df, train_end_point, i, i
        ) == 0).sum(axis=1).groupby('item_nbr').mean().to_frame('item_count0')

        x['item_{}_count0_mean'.format(i)] = \
            df.join(item_count0)['item_count0'].values

        store_mean = get_timespan(
            df, train_end_point, i, i
        ).mean(axis=1).groupby('store_nbr').mean().to_frame('store_mean')

        x['store_{}_mean'.format(i)] = df.join(store_mean)[
            'store_mean'].values

        store_count0 = (get_timespan(df, train_end_point, i, i) == 0).sum(
            axis=1).groupby('store_nbr').mean().to_frame('store_count0')

        x['store_{}_count0_mean'.format(i)] = df.join(store_count0)[
            'store_count0'].values

    for i in range(7):
        x['mean_4_dow{}'.format(i)] = get_timespan(
            df, train_end_point, 28 - i, 4, freq='7D'
        ).mean(axis=1).values

        x['mean_10_dow{}'.format(i)] = get_timespan(
            df, train_end_point, 70 - i, 10, freq='7D'
        ).mean(axis=1).values

        x['count0_10_dow{}'.format(i)] = (get_timespan(
            df, train_end_point, 70 - i, 10, freq='7D'
        ) == 0).sum(axis=1).values

        x['promo_10_dow{}'.format(i)] = get_timespan(
            promo_df, train_end_point, 70 - i, 10, freq='7D'
        ).sum(axis=1).values

        item_mean = get_timespan(
            df, train_end_point, 70 - i, 10, freq='7D'
        ).mean(axis=1).groupby('item_nbr').mean().to_frame('item_mean')
        x['item_mean_10_dow{}'.format(i)] = \
            df.join(item_mean)['item_mean'].values

        x['mean_20_dow{}'.format(i)] = get_timespan(
            df, train_end_point, 140 - i, 20, freq='7D'
        ).mean(axis=1).values

    for i in range(16):
        x["promo_{}".format(i)] = promo_df[
            train_end_point + timedelta(days=i)].values

    if one_hot:
        family_dummy = pd.get_dummies(df.join(items)['family'],
                                      prefix='family')
        x = pd.concat([x, family_dummy.reset_index(drop=True)],
                      axis=1)
        store_dummy = pd.get_dummies(df.reset_index().store_nbr,
                                     prefix='store')
        x = pd.concat([x, store_dummy.reset_index(drop=True)],
                      axis=1)
    else:
        df_items = df.join(items)
        df_stores = df.join(stores)
        x['family'] = df_items['family'].astype(
            'category').cat.codes.values
        x['perish'] = df_items['perishable'].values
        x['item_class'] = df_items['class'].values
        x['store_nbr'] = df.reset_index().store_nbr.values
        x['store_cluster'] = df_stores['cluster'].values
        x['store_type'] = df_stores['type'].astype(
            'category').cat.codes.values

    if is_train:
        y = df[pd.date_range(train_end_point, periods=16)].values
        return x, y
    else:
        return x


# -------------------------------------------------------------------
#   Prepare Dataset
# -------------------------------------------------------------------
X_l, y_l = [], []
first_pred_start = date(2017, 7, 5)  # first_pred_start
n_range = 14
for i in range(n_range):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(first_pred_start - delta)
    X_l.append(X_tmp)
    y_l.append(y_tmp)

X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l
gc.collect()

X_val, y_val = prepare_dataset(date(2017, 7, 26))
X_test = prepare_dataset(date(2017, 8, 16), is_train=False)

params = {
    'num_leaves': 128,  # 31
    'objective': 'regression',
    'min_data_in_leaf': 300,
    'learning_rate': 0.05,
    'feature_fraction': 0.6,  # 0.8
    'bagging_fraction': 0.6,  # 0.8
    'bagging_freq': 2,
    'metric': 'l2',
    'max_bin': 128,
    'num_threads': CPU,
    'seed': SEED,
}

print("Training and predicting models...")
MAX_ROUNDS = 700
val_pred = []
test_pred = []
best_rounds = []
cate_vars = ['family', 'perish', 'store_nbr', 'store_cluster', 'store_type']
w = (X_val["perish"] * 0.25 + 1) / (X_val["perish"] * 0.25 + 1).mean()

for i in range(16):
    print('\n' + '-' * 50 + f'\nDay {i + 1}:\n')

    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=None)
    dval = lgb.Dataset(
        X_val, label=y_val[:, i], reference=dtrain,
        weight=w,
        categorical_feature=cate_vars)
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], verbose_eval=100)

    print("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True)[:15]))
    best_rounds.append(bst.best_iteration or MAX_ROUNDS)

    val_pred.append(
        bst.predict(X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
    test_pred.append(
        bst.predict(X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))
    gc.collect()

score_l = cal_score(y_val, np.array(val_pred).T)
print(
    f'\nDay all, Day 0-5, Day 6-16 = {score_l[0]}, {score_l[1]}, {score_l[2]}'
)

make_submission(df.index, np.array(test_pred).T,
                f'./sub/lgb/{NAME}_seed{SEED}_{score_l[0]}.csv')
