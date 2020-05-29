# -*- coding: utf-8 -*-

import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from typing import Union
from util.functions import pickle_read, pickle_write
from tqdm.auto import tqdm as tqdm

plt.style.use('ggplot')


# ------------------------------------------------------------------------
# https://www.kaggle.com/dhananjay3/wrmsse-evaluator-with-extra-features
# ------------------------------------------------------------------------

class WRMSSE(object):

    def __init__(self,
                 train_df: pd.DataFrame,
                 valid_df: pd.DataFrame,
                 calendar: pd.DataFrame,
                 prices: pd.DataFrame,
                 tr_weight_columns: list,
                 weights=None):
        """
        intialize and calculate weights
        """
        self.group_ids = (
            'all_id', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )

        self.calendar = calendar.copy()
        self.prices = prices.copy()
        self.train_df = train_df.copy()
        self.valid_df = valid_df.copy()

        # last 28 days, e.g. ['d_1886', ..., 'd_1913']
        # self.tr_weight_columns = self.train_df.iloc[:, -28:].columns.tolist()
        self.tr_weight_columns = tr_weight_columns

        self.train_df['all_id'] = "all"

        # id_columns:
        # ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        self.id_columns = \
            [i for i in self.train_df.columns if not i.startswith('d_')]

        # e.g. ['d_1', ..., 'd_1913']: train
        self.tr_target_columns = \
            [i for i in self.train_df.columns if i.startswith('d_')]

        # e.g. ['d_1914', ..., 'd_1941']: valid or hold-out
        self.val_target_columns = \
            [i for i in self.valid_df.columns if i.startswith('d_')]

        # add id_columns if necessary
        if not all([c in self.valid_df.columns for c in self.id_columns]):
            self.valid_df = pd.concat(
                [self.train_df[self.id_columns], self.valid_df],
                axis=1, sort=False
            )

        # get dataframe with evaluation lvl examples (42840 examples)
        self.train_series = self.trans_30490_to_42840(
            self.train_df, self.tr_target_columns,
            # self.group_ids
        )
        self.valid_series = self.trans_30490_to_42840(
            self.valid_df, self.val_target_columns,
            # self.group_ids
        )

        if weights is None:
            self.weights = self.get_weight_df()
        else:
            self.weights = weights

        self.scale = self.get_scale()

        # self.train_series = None
        # self.train_df = None
        # self.prices = None
        # self.calendar = None

    def get_scale(self) -> np.ndarray:
        """
        scaling factor for each series ignoring starting zeros
        """
        print('calculating denominators...')
        scales = []
        for i in range(len(self.train_series)):
            series = self.train_series.iloc[i].values
            # np.argmax returns 1st True index
            series = series[np.argmax(series != 0):]
            if len(series) > 1:
                scale = ((series[1:] - series[:-1]) ** 2).mean()
            else:
                scale = 0
            scales.append(scale)
        return np.array(scales)

    def get_name(self, i) -> str:
        """
        convert a str or list of strings to unique string
        used for naming each of 42840 series
        """
        if type(i) == str or type(i) == int:
            return str(i)
        else:
            return "--".join(i)

    def get_weight_df(self) -> pd.DataFrame:
        """
        returns weights for each of 42840 series in a dataFrame
        """
        print('calculating weights from last 28 days...')

        # {'d_1': 11101, ... }
        day_to_week = self.calendar.set_index("d")["wm_yr_wk"].to_dict()

        # caluculate (sales * price) values in each combination of id and day
        weight_df = self.train_df[
            ["item_id", "store_id"] + self.tr_weight_columns
            ].set_index(["item_id", "store_id"])
        weight_df = (
            weight_df.stack().reset_index().rename(
                columns={"level_2": "d", 0: "value"})
        )
        weight_df["wm_yr_wk"] = weight_df["d"].map(day_to_week)
        weight_df = weight_df.merge(
            self.prices, how="left", on=["item_id", "store_id", "wm_yr_wk"]
        )
        weight_df["value"] = weight_df["value"] * weight_df["sell_price"]
        weight_df = weight_df.set_index(
            ["item_id", "store_id", "d"]
        ).unstack(level=2)["value"]
        weight_df = weight_df.loc[
                    zip(self.train_df.item_id, self.train_df.store_id), :
                    ].reset_index(drop=True)
        weight_df = pd.concat(
            # [self.train_df[self.id_columns], weight_df],
            [self.train_df[self.id_columns].reset_index(drop=True), weight_df],
            axis=1, sort=False
        )

        weights_map = {}

        for i, group_id in enumerate(self.group_ids):

            lv_weight = weight_df.groupby(group_id)[
                self.tr_weight_columns
            ].sum().sum(axis=1)
            lv_weight = lv_weight / lv_weight.sum()

            for lw in range(len(lv_weight)):
                weights_map[self.get_name(lv_weight.index[lw])] = \
                    np.array([lv_weight.iloc[lw]])

        # noinspection PyTypeChecker
        weights = pd.DataFrame(weights_map).T / len(self.group_ids)  # K=12

        # noinspection PyTypeChecker
        return weights

    def trans_30490_to_42840(self, df, cols, dis=False) -> pd.DataFrame:
        """
        Transform 30490 series to all 42840 series for evaluation metric.
        Even if some product ids are excluded, this can work.

        Group                   | d_x0, d_x1, ..., d_n
        -----------------------------------------------
        all                     |  100,  100, ..., 100
        CA                      |   30,   30, ...,  30
        ...                     |  ...    ...  ...
        HOUSEHOLD_2_516--WI_3   |    0,    1, ...,   1
        """
        series_map = {}
        for i, group_id in enumerate(self.group_ids):
            tr = df.groupby(group_id)[cols].sum()
            for i in range(len(tr)):
                series_map[self.get_name(tr.index[i])] = tr.iloc[i].values
        return pd.DataFrame(series_map).T

    def get_rmsse(self, valid_preds) -> pd.Series:
        """
        returns rmsse scores for all 42840 series
        """
        numerator = ((self.valid_series - valid_preds) ** 2).mean(axis=1)
        rmsse = (numerator / self.scale).map(np.sqrt)
        return rmsse

    def get_score(self, valid_preds: Union[pd.DataFrame]) -> list:
        # assertion
        assert self.valid_df[self.val_target_columns].shape == \
               valid_preds.shape

        self.valid_preds = pd.concat(
            # [self.valid_df[self.id_columns].reset_index(), valid_preds],
            [self.valid_df[self.id_columns].reset_index(drop=True),
             valid_preds],
            axis=1, sort=False
        )

        # noinspection PyTypeChecker
        self.valid_preds = self.trans_30490_to_42840(self.valid_preds,
                                                     self.val_target_columns)
        self.rmsse = self.get_rmsse(self.valid_preds)
        self.contributors = pd.concat(
            [self.weights, self.rmsse], axis=1, sort=False
        ).prod(axis=1)

        # noinspection PyTypeChecker
        return [np.sum(self.contributors[self.contributors != np.inf]),
                self.rmsse, self.weights,
                self.valid_series, self.valid_preds]


def wrmsse_initializer(train, calendar, prices, x_tr,
                       x_val, x_val_date_period):
    train_ini = train.copy()
    cal_ini = calendar.copy()
    price_ini = prices.copy()
    x_tr_ini = x_tr.copy()

    x_tr_ini = x_tr_ini.merge(cal_ini[['date', 'd']], how='left', on='date')
    tr_d_l = x_tr_ini['d'].unique().tolist()
    tr_d_max = max([int(dl.split('_')[1]) for dl in tr_d_l])
    id_l = train_ini.columns[:6].to_list()
    id_l.extend(tr_d_l)
    val_d_l = [f'd_{tr_d_max + p + 1}' for p in range(x_val_date_period)]

    train_ini = train_ini[train_ini.id.isin(x_val.id)]
    valid = train_ini.loc[:, val_d_l]

    tr_weight_columns = x_tr_ini.d.unique()[-28:].tolist()
    e = WRMSSE(train_ini, valid, cal_ini, price_ini, tr_weight_columns)
    gc.collect()
    return e


class WSPL(object):

    def __init__(self,
                 train_df: pd.DataFrame,
                 valid_df: pd.DataFrame,
                 calendar: pd.DataFrame,
                 prices: pd.DataFrame,
                 tr_weight_columns: list,
                 weights=None):
        """
        Weighted Scaled Pinball Loss on M5
        Args:
            tr_weight_columns: list of day names, e.g. [d_XXXX, ...]
        """
        self.group_ids = (
            'all_id', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )

        self.calendar = calendar.copy()
        self.prices = prices.copy()
        self.train_df = train_df.copy()
        self.valid_df = valid_df.copy()

        # last 28 days, e.g. ['d_1886', ..., 'd_1913']
        # self.tr_weight_columns = self.train_df.iloc[:, -28:].columns.tolist()
        self.tr_weight_columns = tr_weight_columns

        self.train_df['all_id'] = "all"

        # id_columns:
        # ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        self.id_columns = \
            [i for i in self.train_df.columns if not i.startswith('d_')]

        # e.g. ['d_1', ..., 'd_1913']: train
        self.tr_target_columns = \
            [i for i in self.train_df.columns if i.startswith('d_')]

        # e.g. ['d_1914', ..., 'd_1941']: valid or hold-out
        self.val_target_columns = \
            [i for i in self.valid_df.columns if i.startswith('d_')]

        # add id_columns if necessary
        if not all([c in self.valid_df.columns for c in self.id_columns]):
            self.valid_df = pd.concat(
                [self.train_df[self.id_columns], self.valid_df],
                axis=1, sort=False
            )

        # get dataframe with evaluation lvl examples (42840 examples)
        self.train_series = self.trans_30490_to_42840(
            self.train_df, self.tr_target_columns,
            # self.group_ids
        )
        self.valid_series = self.trans_30490_to_42840(
            self.valid_df, self.val_target_columns,
            # self.group_ids
        )

        if weights is None:
            self.weights = self.get_weight_df()
        else:
            self.weights = weights

        self.scale = self.get_scale()

        # self.train_series = None
        # self.train_df = None
        # self.prices = None
        # self.calendar = None

    def get_scale(self) -> np.ndarray:
        """
        scaling factor for each series ignoring starting zeros
        """
        print('calculating denominators...')
        scales = []
        for i in range(len(self.train_series)):
            series = self.train_series.iloc[i].values  # np.ndarray
            # np.argmax returns 1st True index
            series = series[np.argmax(series != 0):]
            if len(series) > 1:
                scale = np.abs(series[1:] - series[:-1]).mean()
            else:
                scale = 0
            scales.append(scale)
        return np.array(scales)

    def get_name(self, i) -> str:
        """
        convert a str or list of strings to unique string
        used for naming each of 42840 series
        """
        if type(i) == str or type(i) == int:
            return str(i)
        else:
            return "--".join(i)

    def get_weight_df(self) -> pd.DataFrame:
        """
        returns weights for each of 42840 series in a dataFrame
        """
        print('calculating weights from last 28 days...')

        # {'d_1': 11101, ... }
        day_to_week = self.calendar.set_index("d")["wm_yr_wk"].to_dict()

        # caluculate (sales * price) values in each combination of id and day
        weight_df = self.train_df[
            ["item_id", "store_id"] + self.tr_weight_columns
            ].set_index(["item_id", "store_id"])
        weight_df = (
            weight_df.stack().reset_index().rename(
                columns={"level_2": "d", 0: "value"})
        )
        weight_df["wm_yr_wk"] = weight_df["d"].map(day_to_week)
        weight_df = weight_df.merge(
            self.prices, how="left", on=["item_id", "store_id", "wm_yr_wk"]
        )
        weight_df["value"] = weight_df["value"] * weight_df["sell_price"]
        weight_df = weight_df.set_index(
            ["item_id", "store_id", "d"]
        ).unstack(level=2)["value"]
        weight_df = weight_df.loc[
                    zip(self.train_df.item_id, self.train_df.store_id), :
                    ].reset_index(drop=True)
        weight_df = pd.concat(
            # [self.train_df[self.id_columns], weight_df],
            [self.train_df[self.id_columns].reset_index(drop=True), weight_df],
            axis=1, sort=False
        )

        weights_map = {}

        for i, group_id in enumerate(self.group_ids):

            lv_weight = weight_df.groupby(group_id)[
                self.tr_weight_columns
            ].sum().sum(axis=1)
            lv_weight = lv_weight / lv_weight.sum()

            for lw in range(len(lv_weight)):
                weights_map[self.get_name(lv_weight.index[lw])] = \
                    np.array([lv_weight.iloc[lw]])

        # noinspection PyTypeChecker
        weights = pd.DataFrame(weights_map).T / len(self.group_ids)  # K=12

        # noinspection PyTypeChecker
        return weights

    def trans_30490_to_42840(self, df, cols, dis=False) -> pd.DataFrame:
        """
        Transform 30490 series to all 42840 series for evaluation metric.
        Even if some product ids are excluded, this can work.

        Group                   | d_x0, d_x1, ..., d_n
        -----------------------------------------------
        all                     |  100,  100, ..., 100
        CA                      |   30,   30, ...,  30
        ...                     |  ...    ...  ...
        HOUSEHOLD_2_516--WI_3   |    0,    1, ...,   1
        """
        series_map = {}
        for i, group_id in enumerate(self.group_ids):
            tr = df.groupby(group_id)[cols].sum()
            for i in range(len(tr)):
                series_map[self.get_name(tr.index[i])] = tr.iloc[i].values
        return pd.DataFrame(series_map).T

    def get_spl(self, valid_preds: pd.DataFrame, quantile) -> pd.Series:
        """
        Args:
            DataFrame of predictions with (42840, 28) shape
        Return:
            Series of SPL scores for all 42840 insrances
        """
        indicator_p = ((self.valid_series - valid_preds) >= 0).astype(int)
        indicator_n = ((self.valid_series - valid_preds) < 0).astype(int)

        numerator_left = (self.valid_series - valid_preds) * \
                         quantile * indicator_p
        numerator_right = (valid_preds - self.valid_series) * \
                          (1 - quantile) * indicator_n

        # numerator = ((self.valid_series - valid_preds) ** 2).mean(axis=1)
        numerator = (numerator_left + numerator_right).mean(axis=1)
        spl = numerator / self.scale
        return spl

    def get_score(self, valid_preds: Union[pd.DataFrame], quantile) -> list:
        # assertion
        assert self.valid_df[self.val_target_columns].shape == \
               valid_preds.shape

        self.valid_preds = pd.concat(
            # [self.valid_df[self.id_columns].reset_index(), valid_preds],
            [self.valid_df[self.id_columns].reset_index(drop=True),
             valid_preds],
            axis=1, sort=False
        )

        # noinspection PyTypeChecker
        self.valid_preds = self.trans_30490_to_42840(self.valid_preds,
                                                     self.val_target_columns)
        self.rmsse = self.get_spl(self.valid_preds, quantile)
        self.contributors = pd.concat(
            [self.weights, self.rmsse], axis=1, sort=False
        ).prod(axis=1)

        # noinspection PyTypeChecker
        return [np.sum(self.contributors[self.contributors != np.inf]),
                self.rmsse, self.weights,
                self.valid_series, self.valid_preds]


def wspl_initializer(train, calendar, prices, x_tr,
                     x_val, x_val_date_period):
    train_ini = train.copy()
    cal_ini = calendar.copy()
    price_ini = prices.copy()
    x_tr_ini = x_tr.copy()

    x_tr_ini = x_tr_ini.merge(cal_ini[['date', 'd']], how='left', on='date')
    tr_d_l = x_tr_ini['d'].unique().tolist()
    tr_d_max = max([int(dl.split('_')[1]) for dl in tr_d_l])
    id_l = train_ini.columns[:6].to_list()
    id_l.extend(tr_d_l)
    val_d_l = [f'd_{tr_d_max + p + 1}' for p in range(x_val_date_period)]

    train_ini = train_ini[train_ini.id.isin(x_val.id)]
    valid = train_ini.loc[:, val_d_l]

    tr_weight_columns = x_tr_ini.d.unique()[-28:].tolist()
    wspl = WSPL(train_ini, valid, cal_ini, price_ini, tr_weight_columns)
    gc.collect()
    return wspl


if __name__ == '__main__':
    # val_pred = pd.read_csv('./sub/kernel/submission01_0.15905.csv')
    sub = pd.read_csv('./input/sample_submission_uncertainty.csv')
    val_pred = pd.read_csv('./oof/test-for-wrmsse_fold0.csv')
    train_df = pd.read_csv('./input/sales_train_validation.csv')
    train_df['id'] = train_df.item_id + '_' + train_df.store_id
    calendar = pd.read_csv('./input/calendar.csv')
    prices = pd.read_csv('./input/sell_prices.csv')

    val_pred = val_pred.merge(calendar[['date', 'd']], how='left', on='date')
    val_min_d = int(val_pred.d[0].split('_')[1])
    tr_weight_columns = [f'd_{d}' for d in range(val_min_d - 28, val_min_d)]

    print(train_df.id.nunique(), val_pred.id.nunique())
    train_df = train_df[train_df.id.isin(val_pred.id)]
    valid_fold_df = train_df.loc[:, val_pred.d.unique().tolist()]

    e = WRMSSE(train_df, valid_fold_df, calendar, prices, tr_weight_columns)
    print(e.train_series.shape, e.valid_series.shape, e.weights.shape)
    # del train_fold_df, train_df, calendar, prices

    wspl = WSPL(train_df=train_df,
                valid_df=valid_fold_df,
                calendar=calendar,
                prices=prices,
                tr_weight_columns=tr_weight_columns)

    # ---------------------------------------------------------------
    #   Check WRMSSE and WSPL
    # ---------------------------------------------------------------
    data = pickle_read(f'./input/fe_002.pkl')
    x_val = data[(data['date'] >= val_pred.date.min()) &
                 (data['date'] <= val_pred.date.max())]
    x_val = x_val[x_val.id.isin(val_pred.id)]
    print(f'x_val shape {x_val.shape}\nval_pred shape {val_pred.shape}')

    del data
    gc.collect()

    calendar['date'] = pd.to_datetime(calendar['date'], format='%Y-%m-%d')
    x_val = x_val.merge(calendar[['date', 'd']], how='left', on='date')
    x_val_date = x_val.date.unique()
    rmse = np.sqrt(mean_squared_error(x_val.demand, val_pred.demand.values))
    print(f'RMSE = {rmse:.5f}')
    valid_preds = x_val.copy()
    valid_preds['demand'] = val_pred.demand.values

    valid_preds = valid_preds.loc[:, ['id', 'd', 'demand']].pivot(
        index='id', columns='d', values='demand'
    ).reset_index()
    id_columns = [c for c in train_df.columns if 'id' in c]
    valid_preds = train_df[id_columns].merge(
        valid_preds, how='left', on='id'
    ).drop(id_columns, axis=1)

    # WRMSSE
    oof_wrmsse, oof_rmsse, oof_weights, oof_preds, oof_actuals = \
        e.get_score(valid_preds)
    print(f'WRMSSE on validation = {oof_wrmsse:.5f}')

    # WSPL
    oof_wspl, oof_spl, oof_weights, oof_preds, oof_actuals = \
        wspl.get_score(valid_preds, quantile=0.5)
    print(f'WSPL on validation = {oof_wspl:.5f}')

    keys = ['all', 'CA', 'TX', 'WI']
    fig, ax = plt.subplots(len(keys), 1, figsize=(10, 10))
    ax = ax.flatten()
    for i, k in enumerate(keys):
        ax[i].plot(x_val_date, oof_actuals.loc[k, :], label='actual',
                   marker='.', ms=4, linewidth=1, color='gray')
        ax[i].plot(x_val_date, oof_preds.loc[k, :], label='pred',
                   marker='.', ms=4, linewidth=1, color='deeppink')
        ax[i].tick_params(axis='x', labelsize=5, color='gray')
        ax[i].set_title(k + f' (RMSSE = {oof_rmsse[k]:.5f})',
                        color='gray', fontsize=10)
        if i == 0: ax[i].legend(loc='lower right', framealpha=0, fontsize=7)
    fig.tight_layout()
    # fig.savefig('./fig/oof/example.png', dpi=100)
