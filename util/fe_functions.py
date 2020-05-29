# -------------------------------------------------------------------
#   Helper functions for Feature Engineering
# -------------------------------------------------------------------

import sys, gc

sys.path.append('..')
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import timedelta
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder


# -------------------------------------------------------------------
#   Memory save
# -------------------------------------------------------------------
def reduce_mem_num(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            # c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.uint32)
                elif c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.uint64)
                elif c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float64)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    if verbose: print(
        'Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem)
    )

    return df


def reduce_mem_num2(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            c_abs = max(np.abs(c_min), np.abs(c_max))

            if str(col_type)[:3] == 'int':
                if c_abs < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_abs < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_abs < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.uint32)
                elif c_abs < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_abs < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_abs < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    if verbose: print(
        'Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem)
    )

    return df


# -------------------------------------------------------------------
#   Encoding
# -------------------------------------------------------------------
def le(df, cols):
    for col in cols:
        # Leave NaN as it is.
        le = LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)

    return df


def fillna_and_le(data,
                  nan_features=['event_name_1', 'event_type_1',
                                'event_name_2', 'event_type_2'],
                  cat=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                       'event_name_1', 'event_type_1',
                       'event_name_2', 'event_type_2']):
    for feature in nan_features:
        data[feature].fillna('unknown', inplace=True)

    for feature in cat:
        encoder = LabelEncoder()
        data[feature] = encoder.fit_transform(data[feature])

    return data


# -------------------------------------------------------------------
#   Melt and Merge
# -------------------------------------------------------------------
def melt_and_merge(tr, price, cal, sub, nrows=30490 * 1913, merge=False):
    # melt sales data, get it ready for training
    tr_tmp = pd.melt(
        tr,
        id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        var_name='day',
        value_name='demand'
    )
    tr_tmp = reduce_mem_num(tr_tmp)
    print('melted train =', tr_tmp.shape)

    tr_tmp = tr_tmp.iloc[-nrows:, :]
    print('extracted train =', tr_tmp.shape)

    # seperate test dataframes
    test1_rows = [row for row in sub['id'] if 'validation' in row]
    test2_rows = [row for row in sub['id'] if 'evaluation' in row]
    test1 = sub[sub['id'].isin(test1_rows)]
    test2 = sub[sub['id'].isin(test2_rows)]

    # change column names
    col_names = ['d_{}'.format(i + 1914) for i in range(test1.shape[1] - 1)]
    col_names.insert(0, 'id')
    test1.columns = col_names
    col_names = ['d_{}'.format(i + 1942) for i in range(test1.shape[1] - 1)]
    col_names.insert(0, 'id')
    test2.columns = col_names

    # get product table
    product = tr[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']]

    # merge with product table
    # validation
    test1 = test1.merge(product, how='left', on='id')
    # evaluation
    test2['id'] = test2['id'].str.replace('_evaluation', '_validation')
    test2 = test2.merge(product, how='left', on='id')
    test2['id'] = test2['id'].str.replace('_validation', '_evaluation')

    test1 = pd.melt(
        test1,
        id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        var_name='day', value_name='demand'
    )
    test2 = pd.melt(
        test2,
        id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        var_name='day', value_name='demand'
    )

    tr_tmp['part'] = 'train'
    test1['part'] = 'test1'
    test2['part'] = 'test2'

    # data = pd.concat([tr_tmp, test1, test2], axis=0)
    data = pd.concat([tr_tmp, test1], axis=0)
    print(data.shape)

    del tr_tmp, test1, test2
    gc.collect()

    # drop some calendar features
    cal.drop(['weekday', 'wday', 'month', 'year'], inplace=True, axis=1)

    if merge:
        data = pd.merge(data, cal, how='left', left_on=['day'], right_on=['d'])
        data.drop(['d', 'day'], inplace=True, axis=1)

        data = data.merge(
            price, on=['store_id', 'item_id', 'wm_yr_wk'], how='left'
        )
        print('output dataframe =', data.shape)
    else:
        pass

    gc.collect()

    return data


# -------------------------------------------------------------------
#   Feature Enginnering
# -------------------------------------------------------------------
def get_timespan(df, dt, minus, periods, freq='D'):
    df_date_range = \
        pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)
    return df[df.date.isin(df_date_range)]


def shift(minus, df, dt, period=1,
          freq='D', col_name='demand'):
    ret = get_timespan(df, dt, minus, period, freq)
    return ret.loc[:, ['id', col_name]].reset_index(drop=True)


def rolling_sum(minus, df, dt,
                freq='D', col_name='demand'):
    period = minus
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').sum().reset_index()


def rolling_mean(minus, df, dt,
                 freq='D', col_name='demand'):
    period = minus
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').mean().reset_index()


def rolling_mean_div_cur(minus, df, dt,
                         freq='D', col_name='demand'):
    period = minus
    cur = get_timespan(df, dt, 0, 1, freq)[['id', col_name]].reset_index(
        drop=True
    ).rename(columns={'id': 'id', col_name: 'cur_price'})
    ret = get_timespan(df, dt, minus, period, freq)[['id', col_name]]
    ret = ret.groupby('id').mean().reset_index()
    ret = ret.merge(cur, how='left', on='id')
    ret[col_name] = ret[col_name] / ret['cur_price']
    return ret.drop('cur_price', axis=1)


def rolling_mean_decay(minus, df, dt,
                       freq='D', col_name='demand'):
    period = minus
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').apply(
        lambda x: (x[col_name] * np.power(0.9, np.arange(period))).sum()
    ).reset_index()


def rolling_median(minus, df, dt,
                   freq='D', col_name='demand'):
    period = minus
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').median().reset_index()


def rolling_median_div_cur(minus, df, dt,
                           freq='D', col_name='demand'):
    period = minus
    cur = get_timespan(df, dt, 0, 1, freq)[['id', col_name]].reset_index(
        drop=True
    ).rename(columns={'id': 'id', col_name: 'cur_price'})
    ret = get_timespan(df, dt, minus, period, freq)[['id', col_name]]
    ret = ret.groupby('id').median().reset_index()
    ret = ret.merge(cur, how='left', on='id')
    ret[col_name] = ret[col_name] / ret['cur_price']
    return ret.drop('cur_price', axis=1)


def rolling_min(minus, df, dt,
                freq='D', col_name='demand'):
    period = minus
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').min().reset_index()


def rolling_min_div_cur(minus, df, dt,
                        freq='D', col_name='demand'):
    period = minus
    cur = get_timespan(df, dt, 0, 1, freq)[['id', col_name]].reset_index(
        drop=True
    ).rename(columns={'id': 'id', col_name: 'cur_price'})
    ret = get_timespan(df, dt, minus, period, freq)[['id', col_name]]
    ret = ret.groupby('id').min().reset_index()
    ret = ret.merge(cur, how='left', on='id')
    ret[col_name] = ret[col_name] / ret['cur_price']
    return ret.drop('cur_price', axis=1)


def rolling_max(minus, df, dt,
                freq='D', col_name='demand'):
    period = minus
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').max().reset_index()


def rolling_max_div_cur(minus, df, dt,
                        freq='D', col_name='demand'):
    period = minus
    cur = get_timespan(df, dt, 0, 1, freq)[['id', col_name]].reset_index(
        drop=True
    ).rename(columns={'id': 'id', col_name: 'cur_price'})
    ret = get_timespan(df, dt, minus, period, freq)[['id', col_name]]
    ret = ret.groupby('id').max().reset_index()
    ret = ret.merge(cur, how='left', on='id')
    ret[col_name] = ret[col_name] / ret['cur_price']
    return ret.drop('cur_price', axis=1)


def rolling_std(minus, df, dt,
                freq='D', col_name='demand'):
    period = minus
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').std().reset_index()


def rolling_std_div_cur(minus, df, dt,
                        freq='D', col_name='demand'):
    period = minus
    cur = get_timespan(df, dt, 0, 1, freq)[['id', col_name]].reset_index(
        drop=True
    ).rename(columns={'id': 'id', col_name: 'cur_price'})
    ret = get_timespan(df, dt, minus, period, freq)[['id', col_name]]
    ret = ret.groupby('id').std().reset_index()
    ret = ret.merge(cur, how='left', on='id')
    ret[col_name] = ret[col_name] / ret['cur_price']
    return ret.drop('cur_price', axis=1)


def rolling_cov(minus, df, dt,
                freq='D', col_name='demand'):
    period = minus
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').apply(
        lambda x: x.std() / (x.mean() + 1e-16)
    ).reset_index()


def rolling_skew(minus, df, dt,
                 freq='D', col_name='demand'):
    period = minus
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').skew().reset_index()


def rolling_skew_div_cur(minus, df, dt,
                         freq='D', col_name='demand'):
    period = minus
    cur = get_timespan(df, dt, 0, 1, freq)[['id', col_name]].reset_index(
        drop=True
    ).rename(columns={'id': 'id', col_name: 'cur_price'})
    ret = get_timespan(df, dt, minus, period, freq)[['id', col_name]]
    ret = ret.groupby('id').skew().reset_index()
    ret = ret.merge(cur, how='left', on='id')
    ret[col_name] = ret[col_name] / ret['cur_price']
    return ret.drop('cur_price', axis=1)


def rolling_kurt(minus, df, dt,
                 freq='D', col_name='demand'):
    period = minus
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').apply(
        lambda x: x[col_name].kurt()
    ).reset_index()


def rolling_kurt_div_cur(minus, df, dt,
                         freq='D', col_name='demand'):
    period = minus
    cur = get_timespan(df, dt, 0, 1, freq)[['id', col_name]].reset_index(
        drop=True
    ).rename(columns={'id': 'id', col_name: 'cur_price'})
    ret = get_timespan(df, dt, minus, period, freq)[['id', col_name]]
    ret = ret.groupby('id').kurt().reset_index()
    ret = ret.merge(cur, how='left', on='id')
    ret[col_name] = ret[col_name] / ret['cur_price']
    return ret.drop('cur_price', axis=1)


def rolling_quantile(minus, df, dt, q,
                     freq='D', col_name='demand'):
    period = minus
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').quantile(q=q).reset_index()


def rolling_count0(minus, df, dt,
                   freq='D', col_name='demand'):
    period = minus
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').apply(
        lambda x: (x[col_name] == 0).sum()
    ).reset_index()


def rolling_diff_mean(minus, df, dt,
                      freq='D', col_name='demand'):
    # https://note.nkmk.me/python-pandas-diff-pct-change/
    period = minus
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').apply(
        lambda x: x[col_name].diff().mean()
    ).reset_index()


def rolling_pct_mean(minus, df, dt, period=7,
                     freq='D', col_name='demand'):
    # https://note.nkmk.me/python-pandas-diff-pct-change/
    numerator = rolling_mean(period, df, dt)
    numerator.columns = ['id', 'numerator']
    ret = rolling_mean(minus, df, dt)
    ret['demand'] = numerator['numerator'] / (ret['demand'] + 1e-16)
    ret.loc[ret['demand'] == np.inf, 'demand'] = ret['demand'].max() * 10
    return ret


def rolling_mean_diff(minus, df, dt, period=7,
                      freq='D', col_name='demand'):
    # https://note.nkmk.me/python-pandas-diff-pct-change/
    substracted = rolling_mean(period, df, dt)
    substracted.columns = ['id', 'substracted']
    ret = rolling_mean(minus, df, dt)
    ret['demand'] = substracted['substracted'] - ret['demand']
    return ret


def rolling_mean_period(minus, df, dt, period=28,
                        freq='D', col_name='demand'):
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').mean().reset_index()


def rolling_mean_period_div_cur(minus, df, dt, period=28,
                                freq='D', col_name='demand'):
    cur = get_timespan(df, dt, 0, 1, freq)[['id', col_name]].reset_index(
        drop=True
    ).rename(columns={'id': 'id', col_name: 'cur_price'})
    ret = get_timespan(df, dt, minus, period, freq)[['id', col_name]]
    ret = ret.groupby('id').mean().reset_index()
    ret = ret.merge(cur, how='left', on='id')
    ret[col_name] = ret[col_name] / ret['cur_price']
    return ret.drop('cur_price', axis=1)


def rolling_xxx_mean(minus, df, dt, on,
                     freq='D', col_name='demand'):
    if isinstance(on, str):
        on = [on]
    period = minus
    ret = get_timespan(df, dt, minus, period, freq)
    ret_gb = ret[on + [col_name]].groupby(on).mean().reset_index()
    ret_gb.columns = on + [col_name]
    return ret.loc[~ ret.loc[:, ['id']].duplicated(), ['id'] + on].merge(
        ret_gb, how='left', on=on
    ).loc[:, ['id', col_name]]


def rolling_xxx_quantile(minus, df, dt, on, q,
                         freq='D', col_name='demand'):
    if isinstance(on, str):
        on = [on]
    period = minus
    ret = get_timespan(df, dt, minus, period, freq)
    ret_gb = ret[on + [col_name]].groupby(on).quantile(q=q).reset_index()
    ret_gb.columns = on + [col_name]
    return ret.loc[~ ret.loc[:, ['id']].duplicated(), ['id'] + on].merge(
        ret_gb, how='left', on=on
    ).loc[:, ['id', col_name]]


def rolling_xxx_count0_mean(minus, df, dt, on,
                            freq='D', col_name='demand'):
    if isinstance(on, str):
        on = [on]
    period = minus
    ret = get_timespan(df, dt, minus, period, freq)
    ret_gb = ret[on + [col_name]].groupby(on).apply(
        lambda x: (x[col_name] == 0).mean()
    ).reset_index()
    ret_gb.columns = on + [col_name]
    return ret.loc[~ ret.loc[:, ['id']].duplicated(), ['id'] + on].merge(
        ret_gb, how='left', on=on
    ).loc[:, ['id', col_name]]


def dow_mean(minus, df, dt, period,
             freq='7D', col_name='demand'):
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').mean().reset_index()


def dow_mean_div_cur(minus, df, dt, period,
                     freq='7D', col_name='sell_price'):
    cur = get_timespan(df, dt, 0, 1, freq='D')[['id', col_name]].reset_index(
        drop=True
    ).rename(columns={'id': 'id', col_name: 'cur_price'})
    ret = get_timespan(df, dt, minus, period, freq)[['id', col_name]]
    ret = ret.groupby('id').mean().reset_index()
    ret = ret.merge(cur, how='left', on='id')
    ret[col_name] = ret[col_name] / ret['cur_price']
    return ret.drop('cur_price', axis=1)


def dow_count0(minus, df, dt, period,
               freq='7D', col_name='demand'):
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').apply(
        lambda x: (x[col_name] == 0).sum()
    ).reset_index()


def dow_xxx_mean(minus, df, dt, period, on,
                 freq='7D', col_name='demand'):
    if isinstance(on, str):
        on = [on]
    ret = get_timespan(df, dt, minus, period, freq)
    ret_gb = ret[on + [col_name]].groupby(on).mean().reset_index()
    ret_gb.columns = on + [col_name]
    return ret.loc[~ ret.loc[:, ['id']].duplicated(), ['id'] + on].merge(
        ret_gb, how='left', on=on
    ).loc[:, ['id', col_name]]


def dow_xxx_count0_mean(minus, df, dt, period, on,
                        freq='7D', col_name='demand'):
    if isinstance(on, str):
        on = [on]
    ret = get_timespan(df, dt, minus, period, freq)
    ret_gb = ret[on + [col_name]].groupby(on).apply(
        lambda x: (x[col_name] == 0).mean()
    ).reset_index()
    ret_gb.columns = on + [col_name]
    return ret.loc[~ ret.loc[:, ['id']].duplicated(), ['id'] + on].merge(
        ret_gb, how='left', on=on
    ).loc[:, ['id', col_name]]


def rolling_countNaN(minus, df, dt,
                     freq='D', col_name='sell_price'):
    period = minus
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').apply(
        lambda x: x[col_name].isnull().sum()
    ).reset_index()


# for price pct change
# prices of some products are very low, and make abnormal pct changes.
def p_shift_pct(minus, df, dt,
                freq='D', col_name='sell_price'):
    # https://note.nkmk.me/python-pandas-diff-pct-change/
    period = minus + 1
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').apply(
        lambda x: x[col_name].pct_change(minus).mean()
    ).reset_index()


# for price pct change
def p_rolling_pct_mean(minus, df, dt, pct_change_period,
                       freq='D', col_name='sell_price'):
    # https://note.nkmk.me/python-pandas-diff-pct-change/
    period = minus + 1
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').apply(
        lambda x: x[col_name].pct_change(pct_change_period).mean()
    ).reset_index()


def dom_mean(minus, df, dt, period,
             freq='28D', col_name='demand'):
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').mean().reset_index()


def dom_count0(minus, df, dt, period,
               freq='28D', col_name='demand_0'):
    ret = get_timespan(df, dt, minus, period, freq)
    return ret[['id', col_name]].groupby('id').sum().reset_index()


if __name__ == '__main__':
    # -------------------------------------------------------------------
    #   Set Env
    # -------------------------------------------------------------------
    CPU = 3  # to save memory usage

    DEBUG = True
    # DEBUG = False

    if DEBUG:
        N_ROWS = 30490 * 365
    else:
        N_ROWS = 30490 * 1913

    # -------------------------------------------------------------------
    #   Load Dataset
    # -------------------------------------------------------------------
    tr = pd.read_csv('./input/sales_train_validation.csv').pipe(reduce_mem_num)
    print('train shape =', tr.shape)

    price = pd.read_csv('./input/sell_prices.csv').pipe(reduce_mem_num)
    print('price shape =', price.shape)

    cal = pd.read_csv('./input/calendar.csv').pipe(reduce_mem_num)
    print('calendar shape =', cal.shape)

    sub = pd.read_csv('./input/sample_submission.csv').pipe(reduce_mem_num)
    print('submission shape =', sub.shape)

    # -------------------------------------------------------------------
    #   Label Encoding
    # -------------------------------------------------------------------
    tr = le(
        tr, ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    ).pipe(reduce_mem_num)

    price = le(price, ["item_id", "store_id"]).pipe(reduce_mem_num)

    cal = le(
        cal, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    ).pipe(reduce_mem_num)

    # -------------------------------------------------------------------
    #   Melt and Merge
    # -------------------------------------------------------------------
    data = melt_and_merge(
        tr=tr, price=price, cal=cal, sub=sub, nrows=N_ROWS, merge=True
    )
    data['id'] = data['id'].str.replace('_validation', '')
    print('data shape =', data.shape)
