# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
#   Documentations:
#   - HyperOpt formal page
#   https://github.com/hyperopt/hyperopt/wiki/FMin
#
#   - simple way to retrieve parameter value when using hp.choice
#   https://github.com/hyperopt/hyperopt/issues/384
#
# -------------------------------------------------------------------

import gc
import os
import warnings
import psutil
import numpy as np
import pandas as pd

from pprint import pprint, pformat
from hyperopt import hp, tpe, Trials, space_eval
from hyperopt.fmin import fmin

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer

import xgboost as xgb
import lightgbm as lgb

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

CPU = psutil.cpu_count() - 1
SEED = 2020

df = pd.read_csv('./klib/train.csv', engine='c')
df = df[df['Age'].notnull()]
train_x = df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'],
                  axis=1)
train_x = pd.get_dummies(train_x, drop_first=True)
train_y = df['Survived']


def gini(truth, predictions):
    g = np.asarray(np.c_[truth, predictions, np.arange(len(truth))],
                   dtype=np.float)
    g = g[np.lexsort((g[:, 2], -1 * g[:, 1]))]
    gs = g[:, 0].cumsum().sum() / g[:, 0].sum()
    gs -= (len(truth) + 1) / 2.
    return gs / len(truth)


def gini_xgb(predictions, truth):
    truth = truth.get_label()
    return 'gini', -1.0 * gini(truth, predictions) / gini(truth, truth)


def gini_lgb(truth, predictions):
    score = gini(truth, predictions) / gini(truth, truth)
    return 'gini', score, True


def gini_sklearn(truth, predictions):
    return gini(truth, predictions) / gini(truth, truth)


gini_scorer = make_scorer(gini_sklearn,
                          greater_is_better=True,
                          needs_proba=True)


# -------------------------------------------------------------------
#   Random Forest
# -------------------------------------------------------------------
# In this part we'll tune the random forest classsifier,
# using hyperopt library.
# The hyperopt library has a similar purpose as gridsearch,
# but instead of doing an exhaustive search of the parameter space
# it evaluates a few well-chosen data points
# and then extrapolates the optimal solution based on modeling.
# In practice that means it often needs much fewer iterations to
# find a good solution.

# The important parameters to tune are:
# Number of trees in the forest (n_estimators)
# Tree complexity (max_depth)

# CLASSIFICATION...
def objective(params):
    params = {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth'])
    }
    clf = RandomForestClassifier(n_jobs=3, class_weight='balanced', **params)
    score = cross_val_score(
        clf, train_x, train_y,
        scoring=gini_scorer,
        cv=StratifiedKFold()
    ).mean()
    print("Gini {:.3f} params {}".format(score, params))
    return score


space = {
    'n_estimators': hp.quniform('n_estimators', 25, 500, 25),
    'max_depth': hp.quniform('max_depth', 1, 10, 1)
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials)

space_opt = space_eval(space, trials.argmin)

print(' - ' * 20 + '\nOptimal hyper parameters:\n')
pprint(space_opt)


# REGRESSION...
def objective(params):
    params = {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth'])
    }
    rf = RandomForestRegressor(n_jobs=CPU, **params)
    score = cross_val_score(
        rf, train_x, train_y,
        scoring='neg_mean_squared_error',
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED).split(
            train.drop(['first_active_month', 'card_id', 'target'], axis=1),
            target)).mean()
    print("RMSE : {:.5f}\n params :\n{}".format(np.sqrt(-score), params))
    return np.sqrt(-score)


space = {
    'n_estimators': hp.quniform('n_estimators', 25, 500, 25),
    'max_depth': hp.quniform('max_depth', 1, 10, 1)
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials)
space_opt = space_eval(space, trials.argmin)

print(' - ' * 20 + '\nOptimal hyper parameters:\n')
pprint(space_opt)

rf = RandomForestRegressor(
    n_estimators=int(best['n_estimators']),
    max_depth=int(best['max_depth']),
    n_jobs=CPU,
    random_state=SEED).fit(train_x, train_y)


# -------------------------------------------------------------------
#   XGBoost
# -------------------------------------------------------------------
# Similar to tuning above, now we will tune xgboost parameters using hyperopt!
# Initially fixing the number of trees to 250 and
# learning rate to 0.05 (determined that with a quick experiment)
# - then we can find good values for the other parameters.
# Later we can re-iterate this.

# The most important parameters are:
# Number of trees (n_estimators)
# Learning rate - later trees have less influence (learning_rate)
# Tree complexity (max_depth)
# Gamma - Make individual trees conservative, reduce overfitting
# Column sample per tree - reduce overfitting

def objective(params):
    params = {
        'max_depth': int(params['max_depth']),
        'gamma': "{:.3f}".format(params['gamma']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
    }

    clf = xgb.XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        nthread=3,
        **params
    )

    score = cross_val_score(clf, train_x, train_y, scoring=gini_scorer,
                            cv=StratifiedKFold()).mean()
    print("Gini {:.3f} params {}".format(score, params))
    return score


trials = Trials()
space = {
    'max_depth': hp.quniform('max_depth', 2, 8, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'gamma': hp.uniform('gamma', 0.0, 0.5),
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials)
space_opt = space_eval(space, trials.argmin)

print(' - ' * 20 + '\nOptimal hyper parameters:\n')
pprint(space_opt)


# -------------------------------------------------------------------
#   LightGBM
# -------------------------------------------------------------------
#   API reference:
#   https://lightgbm.readthedocs.io/en/latest/Python-API.html#training-api

#   Parameter reference:
#   https://lightgbm.readthedocs.io/en/latest/Parameters.html

# -------------------------------------------------------------------
#   Classification

def objective(params):
    params = {
        # 'boosting_type': 'gbdt',
        'metric': params['metric'],
        'objective': params['objective'],
        'num_leaves': params['num_leaves'],
        'max_depth': params['max_depth'],
        'learning_rate': params['learning_rate'],
        'min_split_gain': params['min_split_gain'],
        'min_child_weight': params['min_child_weight'],
        'min_child_samples': params['min_child_samples'],
        'subsample': params['subsample'],
        'subsample_freq': params['subsample_freq'],
        'colsample_bytree': params['colsample_bytree'],
        'feature_fraction_bynode': params['feature_fraction_bynode'],
        'reg_alpha': params['reg_alpha'],
        'reg_lambda': params['reg_lambda'],
        'n_jobs': params['n_jobs'],
        'seed': params['seed'],
        'verbose': params['verbose'],
    }

    lgbcv = lgb.cv(
        params=params,
        train_set=lgb.Dataset(train_x, label=train_y),
        num_boost_round=10000,
        folds=StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED).split(train_x, train_y),
        metrics='auc',
        init_model=None,
        feature_name='auto',
        categorical_feature='auto',
        early_stopping_rounds=100,
        verbose_eval=1000,
        show_stdv=True,
        seed=SEED,
        callbacks=None)

    score = lgbcv['auc-mean'][np.argmax(lgbcv['auc-mean'])]
    m = 'AUC {:.3f} @ {}\npar {}'.format(score,
                                         np.argmax(lgbcv['auc-mean']),
                                         params)
    print(m)
    return -score


space = {
    'n_jobs': -1,
    'verbose': -1,
    'seed': SEED,
    'metric': 'auc',
    'objective': 'binary',
    'num_leaves': hp.choice(
        'num_leaves',
        2 ** np.linspace(3, 6, 4, dtype=int),
        # [16],
    ),
    # 'max_depth': hp.quniform('max_depth', 1, 10, 1),
    'max_depth': -1,
    'learning_rate': 0.05,  # 0.1
    'min_split_gain': 0.,  # 0.
    'min_child_weight': 1e-3,  # 1e-3
    'min_child_samples': 20,  # 20
    'subsample': hp.choice('subsample', np.arange(0.3, 1.0, 0.1)),
    'subsample_freq': hp.choice('subsample_freq',
                                np.arange(1, 11, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree',
                                  np.arange(0.3, 1.0, 0.1)),
    'feature_fraction_bynode': 1.0,
    'reg_alpha': hp.choice('reg_alpha', np.arange(0., 1.1, 0.1)),
    'reg_lambda': hp.choice('reg_lambda', np.arange(0., 1.1, 0.1)),
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials)
space_opt = space_eval(space, trials.argmin)

print(' - ' * 20 + '\nOptimal hyper parameters:\n')
pprint(space_opt)


# -------------------------------------------------------------------
#   Regression

def objective(params):
    params = {
        # 'boosting_type': 'gbdt',
        'metric': params['metric'],
        'objective': params['objective'],
        'num_leaves': params['num_leaves'],
        'max_depth': params['max_depth'],
        'learning_rate': params['learning_rate'],
        'min_split_gain': params['min_split_gain'],
        'min_child_weight': params['min_child_weight'],
        'min_child_samples': params['min_child_samples'],
        'subsample': params['subsample'],
        'subsample_freq': params['subsample_freq'],
        'colsample_bytree': params['colsample_bytree'],
        'feature_fraction_bynode': params['feature_fraction_bynode'],
        'reg_alpha': params['reg_alpha'],
        'reg_lambda': params['reg_lambda'],
        'n_jobs': params['n_jobs'],
        'seed': params['seed'],
        'verbose': params['verbose'],
    }

    # kf = KFold(n_splits=5, shuffle=True, random_state=SEED).split(
    #     train.drop(['first_active_month', 'card_id', 'target'], axis=1),
    #     target
    # )

    lgbcv = lgb.cv(
        params=params,
        train_set=lgb.Dataset(train_x, train_y),
        num_boost_round=2000,
        # folds=kf,
        metrics='rmse',
        init_model=None,
        feature_name='auto',
        categorical_feature='auto',
        early_stopping_rounds=50,
        verbose_eval=None,
        show_stdv=True,
        seed=SEED,
        callbacks=None)

    score = lgbcv['rmse-mean'][np.argmin(lgbcv['rmse-mean'])]
    best_nr = np.argmin(lgbcv['rmse-mean'])
    print("score : {:.6f}\nbest_nrounds : {}\nparams : \n{}\n".format(
        score, best_nr, params))
    return score


space = {
    'n_jobs': -1,
    'seed': SEED,
    'metric': 'rmse',
    'objective': 'poisson',
    # 'objective': 'tweedie',
    'num_leaves': hp.choice('num_leaves',
                            2 ** np.linspace(3, 6, 4, dtype=int)),
    # 'num_leaves': 16,
    # 'max_depth': hp.quniform('max_depth', 1, 10, 1),
    'max_depth': -1,
    'learning_rate': 0.05,  # 0.1
    'min_split_gain': 0.,  # 0.
    'min_child_weight': 1e-3,  # 1e-3
    'min_child_samples': 20,  # 20
    'subsample': hp.choice('subsample', np.arange(0.3, 1.0, 0.1)),
    'subsample_freq': hp.choice('subsample_freq',
                                np.arange(1, 11, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree',
                                  np.arange(0.3, 1.0, 0.1)),
    'feature_fraction_bynode': 1.0,
    'reg_alpha': hp.choice('reg_alpha', np.arange(0., 1.1, 0.1)),
    'reg_lambda': hp.choice('reg_lambda', np.arange(0., 1.1, 0.1)),
}

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest,
            max_evals=10, trials=trials)
space_opt = space_eval(space, trials.argmin)

print(' - ' * 20 + '\nOptimal hyper parameters:\n')
pprint(space_opt)

params_opt = {
    # 'boosting': BOOSTING,
    'num_leaves': int(space_opt['num_leaves']),
    'max_depth': int(space_opt['max_depth']),
    'learning_rate': space_opt['learning_rate'],
    'objective': 'regression',
    'min_split_gain': space_opt['min_split_gain'],
    'min_child_weight': space_opt['min_child_weight'],
    'min_child_samples': space_opt['min_child_samples'],
    'subsample': space_opt['subsample'],
    'subsample_freq': space_opt['subsample_freq'],
    'colsample_bytree': '{:.3f}'.format(space_opt['colsample_bytree']),
    'reg_alpha': '{:.3f}'.format(space_opt['reg_alpha']),
    'reg_lambda': '{:.3f}'.format(space_opt['reg_lambda']),
    'nthread': CPU,
    'verbose': -1,
}

lgbcv = lgb.cv(
    params=params_opt,
    train_set=lgb.Dataset(train_x, train_y),
    num_boost_round=2000,
    folds=KFold(n_splits=5, shuffle=True, random_state=SEED).split(train_x,
                                                                   train_y),
    metrics='rmse',
    init_model=None,
    feature_name='auto',
    categorical_feature='auto',
    early_stopping_rounds=50,
    verbose_eval=None,
    show_stdv=True,
    seed=SEED,
    callbacks=None)
score = lgbcv['rmse-mean'][np.argmin(lgbcv['rmse-mean'])]
best_nr = np.argmin(lgbcv['rmse-mean'])
print('best nrounds : {}'.format(best_nr))

# -------------------------------------------------------------------
# MODEL SELECTION...
#
rf_model = RandomForestClassifier(
    n_jobs=3,
    class_weight='balanced',
    n_estimators=325,
    max_depth=5
)

xgb_model = xgb.XGBClassifier(
    n_estimators=250,
    learning_rate=0.05,
    nthread=3,
    max_depth=2,
    colsample_bytree=0.7,
    gamma=0.15
)

lgbm_model = lgbm.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.01,
    num_leaves=16,
    colsample_bytree=0.7
)

models = [
    ('Random Forest', rf_model),
    ('XGBoost', xgb_model),
    ('LightGBM', lgbm_model),
]

for label, model in models:
    scores = cross_val_score(model, train_x, train_y, cv=StratifiedKFold(),
                             scoring=gini_scorer)
    print("Gini coefficient: %0.4f (+/- %0.4f) [%s]" % (
        scores.mean(), scores.std(), label))
