# -*- coding: utf-8 -*-
import gc
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from klib.line_notify_1080Ti import notifier
from klib.logger import get_logger
from klib.target_encode import TargetEncoder

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 100)
# dir(pd.options.display)
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')
pd.options.mode.chained_assignment = None
gc.enable()

FI = True
CPU = 2
FEATURE_SAVE = False
FI_DROP = False
ADV = True
DEBUG = False
BOOSTING = 'gbdt'
TH = 1

code_name = 'stacker_v1'

SEED = 71
SEED_LIST = [SEED, 72, 73, 74, 75, 76]
SEL_COL = [
    'var1',
    'var2',
]
META_FLAG = {
    'LGB': False,
}

logger = get_logger('./log/stacking.log')
logger.info('\n' + '-' * 40 + '\n{}\n'.format(code_name) + '-' * 40)
notifier('\n' + '-' * 40 + '\n{}\n'.format(code_name) + '-' * 40)

train_x_name = './input/train_x.f'
test_x_name = './input/test_x.f'

# TARGET and ID
train = pd.read_feather('./input/train.f')
test = pd.read_feather('./input/test.f')
train_y = train.TARGET
train_id = train[['ID']]
test_id = test[['ID']]
del train, test
gc.collect()

# FEATURES
train_x = pd.read_feather(train_x_name)
test_x = pd.read_feather(test_x_name)
train_x = train_x[SEL_COL]
test_x = test_x[SEL_COL]
logger.info(train_x.shape, test_x.shape)
print(train_x.shape, test_x.shape)

# MISSING
# train_x.replace(np.inf, np.nan, inplace=True)
# train_x.replace(-np.inf, np.nan, inplace=True)
# test_x.replace(np.inf, np.nan, inplace=True)
# test_x.replace(-np.inf, np.nan, inplace=True)

# 1ST LAYER
logger.info('\nmeta_fe loading')
print('\nmeta_fe loading')
lgb1_train = pd.read_feather('sub/stack/OOF/' + 'meta' + '.f')
lgb1_test = pd.read_csv('sub/stack/OOF/' + 'meta' + '.f')
if META_FLAG['LGB']:
    train_x.loc[:, 'LGB'] = \
        lgb1_train.loc[:, 'TARGET']
    test_x.loc[:, 'LGB'] = \
        lgb1_test.loc[:, 'TARGET']
print('\n1st layer train_x.shape : {}'.format(train_x.shape[1]))

# HYPER PARAMETER TUNE
print('\nLGB w/ Early Stopping')
params_list = []
cv_result_list = []
for i in [8]:
    for j in [5]:
        params = {
            'boosting': BOOSTING,
            'num_leaves': i,  # ini 63 => cur 16 => drop
            'max_depth': j,  # ini 6 => cur 4 => drop
            'learning_rate': 0.02,  # 0.02
            'objective': 'binary',
            'class_weight': None,
            'min_split_gain': 0.02,  # 0.02
            'min_child_weight': 10,  # ini 10
            'min_child_samples': 150,  # ini 150
            'subsample': 0.9,  # ini 0.9 => cur 0.9 => drop
            'subsample_freq': 1,  # 1
            'colsample_bytree': 0.9,  # ini 0.9 => cur 0.4 => drop
            'reg_alpha': 0.1,  # 0.5
            'reg_lambda': 0.1,  # 0.5
            'scale_pos_weight': 1,  # 1
            'nthread': CPU,
            'verbose': -1,
        }

        lgbcv = lgb.cv(
            params=params,
            train_set=lgb.Dataset(train_x, label=train_y),
            num_boost_round=5000,
            folds=StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=SEED).split(train_x, train_y),
            metrics='auc',
            init_model=None,
            feature_name='auto',
            categorical_feature='auto',
            early_stopping_rounds=100,
            verbose_eval=50,
            show_stdv=True,
            seed=SEED,
            callbacks=None)
        print('\n CV result')
        lgbcv_best = lgbcv['auc-mean'][np.argmax(lgbcv['auc-mean'])]
        lgbcv_best_std = lgbcv['auc-stdv'][np.argmax(lgbcv['auc-mean'])]
        best_round = np.argmax(lgbcv['auc-mean'])
        print('Hyper parameter : ', i, j)
        print('Best AUC : {:.6f} + {:.6f}'.format(lgbcv_best, lgbcv_best_std))
        print('Early stopped @ ', best_round, '\n\n')
        message = 'LGBCV-has-processed\nbest-rounds {}\n{}'.format(
            best_round, lgbcv_best)
        notifier(message=message)
        notifier(message=[i + ':' + str(params[i]) for i in params.keys()])
        params_list.append(params)
        cv_result_list.append(best_round)

# IMPORTANCE
# if FI:
#     params = params
#     lgbt = lgb.train(
#         params=params,
#         train_set=lgb.Dataset(train_x, label=train_y),
#         # early_stopping_rounds=100,
#         num_boost_round=best_round,
#         # num_boost_round=1316,
#         callbacks=None,
#         verbose_eval=False
#     )
#     lgbt.save_model('./model/{}.lgb'.format(
#         code_name + '_' + str(datetime.now()).split(' ')[0]))
#
#     fi_df_s, fi_df_g = lgbm_plot_importance(
#         lgbt,
#         './sub/LGBM/FI/FI_{}_split.csv'.format(
#             code_name + '_' +
#             str(datetime.now()).split(' ')[0] + '_shape' +
#             str(train_x.shape[1])),
#         './sub/LGBM/FI/FI_{}_gain.csv'.format(
#             code_name + '_' +
#             str(datetime.now()).split(' ')[0] + '_shape' +
#             str(train_x.shape[1])),
#         './fig/FI_{}.png'.format(
#             code_name + '_' +
#             str(datetime.now()).split(' ')[0] + '_shape' +
#             str(train_x.shape[1]))
#     )
#     print('null imporance features : ', (fi_df_g.FI == 0).sum())

# NULL IMPORTANCE
if FI_DROP:
    train_x.drop(fi_df_s.name[fi_df_s.FI < TH], axis=1, inplace=True)
    test_x.drop(fi_df_s.name[fi_df_s.FI < TH], axis=1, inplace=True)
    print(train_x.shape)
    best_round = None  # TODO > rerun cv

# TRAIN AND PREDICT
kfold = 5
pred_df = test_id['ID'].to_frame()
oob_train = train_id['ID'].to_frame()
lgbm_cv_score = []
sub = test_id['ID'].to_frame()
lgbm_adv_cv_score = []

for s in SEED_LIST:
    target_label = 'TARGET_{}'.format(s)
    oob_train[target_label] = 0.
    sub[target_label] = 0.

for j, s in enumerate(SEED_LIST):
    print('\n\n {} th loop (SEED:{})'.format(j + 1, s))
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=s)
    target_label = 'TARGET_{}'.format(s)

    for i, (train_index, valid_index) in enumerate(
            skf.split(train_x, train_y)):
        print('-' * 80)
        print('train_index : ', train_index[:10])
        print(' lgb kfold: {}  of  {} : '.format(i + 1, kfold))

        if META_FLAG['LGBM1']:
            train_x.loc[:, 'META_FEATURE_LGBM1'] = \
                lgb1_train.loc[:, 'TARGET_{}'.format(s)]
            test_x.loc[:, 'META_FEATURE_LGBM1'] = \
                lgb1_test.loc[:, 'fold{}_seed{}'.format(i + 1, s)]

        X = train_x.values
        y = train_y.values
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        params_pred = params
        params_pred['seed'] = s
        lgbm_kfold_model = lgb.train(
            params=params_pred,
            train_set=lgb.Dataset(
                X_train, label=y_train),
            num_boost_round=best_round,
            # num_boost_round=1316,
            # early_stopping_rounds=100,
            callbacks=None,
            verbose_eval=False)
        # test
        pred_test = lgbm_kfold_model.predict(test_x.values)
        sub[target_label] += (pd.Series(pred_test).rank() / len(
            pred_test)).values / kfold
        pred_df = pd.concat((pred_df, pd.DataFrame(pred_test)), axis=1)

        # train
        pred_valid = pd.DataFrame(
            lgbm_kfold_model.predict(X_valid),
            index=valid_index)
        oob_train.loc[valid_index, target_label] = pred_valid.values
        lgbm_cv_score.append(roc_auc_score(y_valid, pred_valid.values))

    print('valid AUC  : {:.5f}'.format(
        roc_auc_score(train_y, oob_train[target_label])))
    print('adversarial AUC : {:.5f} + {:.5f}'.format(
        np.mean(lgbm_adv_cv_score[j:j + 6]),
        np.std(lgbm_adv_cv_score[j:j + 6])))
    notifier('{} th loop mean AUC : {:.5f} / valid AUC : {:.5f}'.format(
        j + 1, np.mean(lgbm_cv_score),
        roc_auc_score(train_y, oob_train[target_label])))
    notifier('adversarial AUC : {:.5f} + {:.5f}'.format(
        np.mean(lgbm_adv_cv_score[j:j + 6]),
        np.std(lgbm_adv_cv_score[j:j + 6])))
print('\n\nmeta feature AUC : {:.5f}'.format(
    roc_auc_score(train_y, oob_train.iloc[:, 2:].mean(axis=1))))
print('adv mean  AUC : {:.5f}'.format(np.mean(lgbm_adv_cv_score)))

# SAVE
# TEST
score = roc_auc_score(train_y, oob_train.iloc[:, 2:].mean(axis=1))
score5 = roc_auc_score(train_y, oob_train.iloc[:, 2:7].mean(axis=1))
adv_score = np.mean(lgbm_adv_cv_score)
pred_df.to_csv(
    'sub/LGBM/' + code_name + '_{:.5f}CV_{:.5f}advCV_{}fold_{}loop.csv'.format(
        score, adv_score, kfold, len(SEED_LIST)),
    index=False,
    header=['SK_ID_CURR'] + \
           ['fold{}_'.format(i + 1) + 'seed{}'.format(s) for s in
            SEED_LIST for i in range(kfold)]
)

ave_sub = test_id['SK_ID_CURR'].to_frame()
ave_sub['TARGET'] = sub.iloc[:, 2:].mean(axis=1)
ave_sub.to_csv(
    'sub/LGBM/' + code_name +
    '_{:.5f}CV_{:.5f}advCV_{}SEED_Rank_{}LoopAve.csv'.format(
        score, adv_score, SEED, len(SEED_LIST) - 1),
    index=False)

# OOF
oob_train.to_csv('sub/LGBM/stacking/' + code_name +
                 '_{:.5f}CV_{:.5f}advCV_{}SEED.csv'.format(
                     score, adv_score, SEED), index=False)

for s in SEED_LIST:
    target_label = 'TARGET_{}'.format(s)
    oob_train[target_label] = oob_train[target_label].rank() / len(oob_train)

oob_train.to_csv(
    'sub/LGBM/stacking/' + code_name +
    '_{:.5f}CV_{:.5f}advCV_{}SEED_Rank.csv'.format(
        score, adv_score, SEED), index=False)

print('{} : All done\n5 SEEDs Rank blended AUC : {:.5f}'.format(
    code_name, score5))
print('{} : All done\n8 SEEDs Rank blended AUC : {:.5f}'.format(
    code_name, score))
notifier('{} : All done\n5 SEEDs Rank blended AUC : {:.5f}'.format(
    code_name, score5))
notifier('{} : All done\n8 SEEDs Rank blended AUC : {:.5f}'.format(
    code_name, score))
print('\nAll done !')
