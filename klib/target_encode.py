# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import gc


class TargetEncoder(object):
    def __init__(self, min_sample_leaf=1, smoothing=1, noise_level=0):
        self.min_sample_leaf = min_sample_leaf
        self.smoothing = smoothing
        self.noise_level = noise_level

    def __str__(self):
        return 'Target Encoder'

    @staticmethod
    def add_noise(series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))

    def encode(self,
               train_series=None,
               test_series=None,
               target=None):
        """
        trn_series : training categorical feature as a pd.Series
        tst_series : test categorical feature as a pd.Series
        target : target data as a pd.Series
        min_samples_leaf (int) :
            minimum samples to take category average into account
        smoothing (int) :
            smoothing effect to balance categorical average vs prior
        """
        assert len(train_series) == len(target)
        assert train_series.name == test_series.name

        print(' -' * 10)
        print(' {0} encoded as {1}. \n ... ... ... \n'.format(
            train_series.name, target.name))
        temp = pd.concat([train_series, target], axis=1)

        # mean and count in each value
        averages = temp.groupby(by=train_series.name)[target.name].agg(
            ['mean', 'count'])

        # smoothing ( reliability )
        smoothing = 1 / (
                1 + np.exp(-(averages['count'] - self.min_sample_leaf) /
                           self.smoothing))

        # ground mean
        prior = target.mean()

        # weighted mean via reliability
        averages[target.name] = prior * (1 - smoothing) + averages[
            'mean'] * smoothing

        # drop mean and count columns
        averages.drop(['mean', 'count'], axis=1, inplace=True)

        # apply averages to train and test series
        # -------------------------------------------------------------------------
        # train
        te_train_series = pd.merge(
            train_series.to_frame(train_series.name),
            averages.reset_index().rename(
                columns={target.name: 'average'}),
            on=train_series.name,
            how='left')['average'].rename(train_series.name + '_mean').fillna(
            prior)

        # pd.merge does not keep the index so restore it
        te_train_series.index = train_series.index
        print(' train data encoded.')

        # -------------------------------------------------------------------------
        # test
        te_test_series = pd.merge(
            test_series.to_frame(test_series.name),
            averages.reset_index().rename(
                columns={target.name: 'average'}),
            on=test_series.name,
            how='left')['average'].rename(train_series.name + '_mean').fillna(
            prior)

        # pd.merge does not keep the index so restore it
        te_test_series.index = test_series.index
        print(' test data encoded.\n' + ' -' * 10)

        del temp, averages
        gc.collect()

        return self.add_noise(te_train_series, self.noise_level), \
               self.add_noise(te_test_series, self.noise_level)


if __name__ == '__main__':
    te = TargetEncoder()
    print(te)


# -----------------------------------------------
## Under construction
def target_encode_dask(train_series_dask=None,
                       test_series_dask=None,
                       target_dask=None,
                       min_samples_leaf=1,
                       smoothing=1,
                       noise_level=0,
                       check_length=True,
                       is_dask=True):
    import dask.dataframe as dd

    if check_length:
        print('\nchecking length\n ... ... ...')
        assert len(train_series_dask) == len(target_dask)
        assert train_series_dask.name == test_series_dask.name
        print('\nfinished checking length w/o error.\n')
    else:
        pass

    divisions = train_series_dask.divisions
    temp_dd = dd.multi.concat([train_series_dask, target_dask],
                              axis=1,
                              interleave_partitions=True)
    print(temp_dd.divisions)

    # mean and count in each value
    averages_dd = temp_dd.groupby(by=train_series_dask.name)[
        target_dask.name].agg(['mean', 'count'])
    print(averages_dd.divisions)

    # smoothing ( reliability )
    smoothing_dd = 1 / (
            1 + np.exp(-(averages_dd['count'] - min_samples_leaf) / smoothing))

    # ground mean
    prior_dd = target_dask.mean().compute()

    # weighted mean via reliability (target encoding)
    averages_dd[target_dask.name] = prior_dd * (1 - smoothing_dd) + averages_dd[
        'mean'] * smoothing_dd
    print(averages_dd.divisions)

    # drop mean and count columns
    averages_dd = averages_dd.drop(['mean', 'count'], axis=1)
    print(averages_dd.divisions)

    # apply averages to train and test series
    # -------------------------------------------------------------------------
    # train
    # te_train_series = pd.merge(
    #     train_series.to_frame(train_series.name),
    #     averages.reset_index().rename(
    #         columns={'index': target.name, target.name: 'average'}),
    #     on=train_series.name,
    #     how='left')['average'].rename(train_series.name + '_mean').fillna(prior)

    te_train_dd = dd.merge(
        train_series_dask.to_frame(train_series_dask.name),
        averages_dd.reset_index().rename(columns={target_dask.name: 'ave'}),
        on=train_series_dask.name,
        how='left'
    )['ave'].rename(train_series_dask.name + '_te').fillna(prior_dd)

    return te_train_dd

    # # pd.merge does not keep the index so restore it
    # te_train_series.index = train_series.index
    #
    # # -------------------------------------------------------------------------
    # # test
    # te_test_series = pd.merge(
    #     test_series.to_frame(test_series.name),
    #     averages.reset_index().rename(
    #         columns={'index': target.name, target.name: 'average'}),
    #     on=test_series.name,
    #     how='left')['average'].rename(train_series.name + '_mean').fillna(prior)
    #
    # # pd.merge does not keep the index so restore it
    # te_test_series.index = test_series.index
    # et = time()
    # print('-' * 50 + '\n\n')
    # print('elapsed time : {} m'.format(round((et - st) / 60, 0)))

    # if is_dask:
    #     return add_noise(te_train_series, noise_level), \
    #            add_noise(te_test_series, noise_level)
    # else:
    #     return add_noise(te_train_series, noise_level), \
    #            add_noise(te_test_series, noise_level)
