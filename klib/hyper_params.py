# -*- coding: utf-8 -*-
import psutil
import warnings
import pandas as pd
import numpy as np

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from sklearn.datasets import load_boston

warnings.simplefilter(action='ignore', category=FutureWarning)

CPU = psutil.cpu_count() - 1
SEED = 2020
IS_CLASSIFICATION = True

if not IS_CLASSIFICATION:
    boston = load_boston()
    train_x, train_y = boston.data, boston.target

if IS_CLASSIFICATION:
    # https://www.kaggle.com/c/titanic/data
    train_titanic = pd.read_csv('./klib/train.csv', index_col=0)
    age_imp_map = train_titanic.groupby('Parch')['Age'].median()
    train_x = train_titanic.drop(
        ['Survived', 'Name', 'Ticket', 'Cabin'],
        axis=1)
    train_y = train_titanic.Survived
    for i in age_imp_map.index:
        train_x.loc[(train_x.Age.isnull() & (train_x.Parch == i)), 'Age'] = \
            age_imp_map[i]
    train_x.loc[train_x.Embarked.isnull(),
                'Embarked'] = train_x.Embarked.mode().values[0]
    for i in ['Pclass', 'SibSp', 'Parch', 'Embarked']:
        train_x.loc[:, i] = train_x[i].astype(object)
    train_x = pd.get_dummies(train_x)

# scikit-learn LogisticRegressor.....................................
# http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix

param_grid = {'penalty': ['l1', 'l2'],
              'C': [0.01, 0.1, 10, 100, 1000]}
gridCV_logit = GridSearchCV(
    LogisticRegression(class_weight='balanced',
                       solver='liblinear',
                       random_state=2018,
                       n_jobs=1),
    param_grid=param_grid,
    scoring='roc_auc',
    n_jobs=3,  # memory problem
    cv=5
)
scaler = StandardScaler()  # for L1 penalty
gridCV_logit.fit(scaler.fit_transform(train_x.values), train_y)
logit_best_params = gridCV_logit.best_params_
logti_cv_results = pd.DataFrame(gridCV_logit.cv_results_)
print(logit_best_params, 'Gini : %.5f' % ((gridCV_logit.best_score_ - 0.5) * 2))

# -------------------------------------------------------------------
#   LightGBM
# -------------------------------------------------------------------
#   API reference:
#   https://lightgbm.readthedocs.io/en/latest/Python-API.html#training-api

#   Parameter reference:
#   https://lightgbm.readthedocs.io/en/latest/Parameters.html

params = {
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html#boosting
    'boosting': 'gbdt',  # gbdt(=default),  rf, dart, goss,
    'nthread': CPU,
    'learning_rate': 0.02,  # 0.1
    'num_leaves': 31,  # num_leaves < 2**(max_depth + 1)
    'max_depth': -1,  # -1
    'min_child_weight': 1e-3,  # minimal sum hessian in one leaf, 1e-3
    'min_child_samples': 20,  # minimal number of data in one leaf, 20
    'min_split_gain': 0.,  # minimum loss reduction
    # 'max_delta_step': 0,
    # ---------------------------------------------------------
    # randomly select part of data without resampling, 1.0
    # Note: to enable bagging,
    # `subsample_freq` should be set to a non zero value as well.
    'subsample': 1.0,
    # Note: to enable bagging,
    # `subsample` should be set to value smaller than 1.0 as well.
    'subsample_freq': 1,  # bagging at every k iteration, 1
    # 'bagging_fraction_seed': 3,  # random seed for bagging, default = 3
    # ---------------------------------------------------------
    'colsample_bytree': 0.7,  # part of features on each iteration, 1
    'feature_fraction_bynode': 1.0,  # select X% of features at each tree node
    # ---------------------------------------------------------
    'reg_lambda': 0.1,  # L2 regularization term on weights, 0.
    'reg_alpha': 0.1,  # L1 regularization term on weights, 0.
    # 'sketch_eps': 0.03,
    'scale_pos_weight': 1,
    # 'updater': 'grow_colmaker,prune',
    # 'refresh_leaf': 1,
    # 'process_type': 'default',
    'max_bin': 255,  # 255
    'min_data_in_bin': 3,  # 3
    'subsample_for_bin': 200000,  # bin_construct_sample_cnt, 200000
    # 'predictor': 'cpu_predictor',
    'objective': 'binary',
    'eval_metric': 'auc',
    'verbose': 0,  # < 0: Fatal, = 0: Error (Warning), = 1: Info, > 1: Debug
}

lgbcv = lgb.cv(params=params,
               train_set=lgb.Dataset(train_x, label=train_y),
               num_boost_round=2000,
               folds=None,
               nfold=5,
               stratified=True,
               shuffle=True,
               # metrics=(),  # Evaluation metrics to be watched in CV (string).
               # fobj=None,  # Customized objective function.
               # feval=None,  # Customized evaluation function.
               init_model=None,
               feature_name='auto',
               categorical_feature='auto',
               early_stopping_rounds=50,
               # fpreproc=None,
               # verbose_eval=10,
               show_stdv=True,
               seed=SEED,
               callbacks=None)
print(pd.DataFrame(lgbcv))

model = lgb.train(params=params,
                  train_set=lgb.Dataset(train_x, label=train_y),
                  num_boost_round=2000,
                  valid_sets=lgb.Dataset(valid_x, label=valid_y),
                  # metrics=(),  # Evaluation metrics to be watched in CV (string).
                  # fobj=None,  # Customized objective function.
                  # feval=None,  # Customized evaluation function.
                  init_model=None,
                  feature_name='auto',
                  categorical_feature='auto',
                  early_stopping_rounds=100,
                  # fpreproc=None,
                  verbose_eval=10,
                  keep_training_booster=False,
                  callbacks=None)
dm = model.dump_model()
dm_ti = dm['tree_info']
num_leaves_l = []
for i in dm_ti:
    num_leaves_l.append(i['num_leaves'])

# scikit-learn LGBMRegressor
# https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api
# https://lightgbm.readthedocs.io/en/latest/Parameters.html
# https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

params = {
    'num_leaves': [31],  # 1. num_leaves < 2**(max_depth+1)
    'max_depth': [-1],  # 3
    'learning_rate': [0.1],
    'n_estimators': [100],  # num_iterations
    'max_bin': [255],
    # max number of bins that feature values will be bucketed in.
    # Small number of bins may reduce training accuracy
    # but may increase general power (deal with over-fitting)
    'subsample_for_bin': [200000],  # bin_construct_sample_cnt
    # number of data that sampled to construct histogram bins
    # will give better training result when set this larger,
    # but will increase data loading time.
    # set this to larger value if data is "very sparse".
    'objective': ['binary'],
    'class_weight': [None],
    # minimum loss reduction required to make a further partition on
    # a leaf node of the tree.
    'min_split_gain': 0.,
    'min_child_weight': [1e-3],  # min_sum_hessian_in_leaf
    'min_child_samples': [20],  # min_data_in_leaf
    # ---------------------------------------------------------
    # randomly select part of data without resampling, 1.0
    # Note: to enable bagging,
    # `subsample_freq` should be set to a non zero value as well.
    'subsample': 1.0,
    # Note: to enable bagging,
    # `subsample` should be set to value smaller than 1.0 as well.
    'subsample_freq': 0,  # bagging at every k iteration, 0
    # ---------------------------------------------------------
    'colsample_bytree': 1,  # feature_fraction
    'reg_alpha': [0],  # lambda_l1
    'reg_lambda': [0],  # lambda_l2
    'random_state': [SEED],
    'n_jobs': [3],
}

'''
1. num_leaves < 2**(max_depth)
    When number of leaves are the same,
    the leaf-wise tree is much deeper than depth-wise tree.
    As a result, it may be over-fitting.
2. min_child_samples
    Value depends on the number of training data and num_leaves.
    Setting it to a large value can avoid growing too deep a tree,
    but may cause under-fitting.
    In practice, setting it to hundreds or thousands is enough for
    a large dataset.
3. max_depth
    can use max_depth to limit the tree depth explicitly.
'''

# XGBoost xgb.train/xgb.cv/..........................................
# http://xgboost.readthedocs.io/en/latest//parameter.html

# Cross Validated Procedure
# 1. number of trees
# 2. max_depth and min_child_weight
# 3. gamma
# 4. subsample colsample_bytree
# 5. lambda and alpha
# 6. learning rate and number of trees with early stopping

params = {'booster': 'gbtree',  # gbtree, gblinear or dart
          'silent': 0,  # 0:printing mode 1:silent mode.
          'nthread': 3,
          'eta': 0.1,
          'gamma': 0,
          'max_depth': 10,
          'min_child_weight': 100,
          'max_delta_step': 0,
          # Maximum delta step we allow each tree’s weight estimation to be.
          # If the value is set to 0, it means there is no constraint.
          # If it is set to a positive value,
          # it can help making the update step more conservative.
          # Usually this parameter is not needed,
          # but it might help in logistic regression
          # when class is extremely imbalanced.
          # Set it to value of 1-10 might help control the update.
          'subsample': 0.9,
          'colsample_bytree': 0.9,
          'colsample_bylevel': 0.9,
          'lambda': 1,  # L2 regularization term on weights.
          'alpha': 0,  # L1 regularization term on weights.
          'tree_method': 'auto',
          # The tree construction algorithm used in XGBoost.
          # Distributed and external memory version only support
          # approximate algorithm.
          # Choices:
          # {‘auto’:Use heuristic to choose faster one,
          # ‘exact’:Exact greedy algorithm,
          # ‘approx’:Approximate greedy algorithm
          #         using sketching and histogram,
          # ‘hist’:Fast histogram optimized approximate greedy algorithm,
          # ‘gpu_exact’,
          # ‘gpu_hist’}
          'sketch_eps': 0.03,
          # This is only used for approximate greedy algorithm.
          # This roughly translated into O(1 / sketch_eps) number of bins.
          # Compared to directly select number of bins,
          # this comes with theoretical guarantee with sketch accuracy.
          'scale_pos_weight': 1,
          'updater': 'grow_colmaker,prune',
          'refresh_leaf': 1,
          'process_type': 'default',
          'grow_policy': 'depthwise',
          'max_leaves': 0,
          'max_bin': 256,
          'predictor': 'cpu_predictor',
          'objective': 'binary:logistic',
          # “reg:linear” –linear regression
          # “reg:logistic” –logistic regression
          # “binary:logistic” –logistic regression for binary classification,
          #         output probability
          # “binary:logitraw” –logistic regression for binary classification,
          #         output score before logistic transformation
          # “count:poisson” –poisson regression for count data,
          #         output mean of poisson distribution
          #         * max_delta_step is set to 0.7 by default
          #         in poisson regression (used to safeguard optimization)
          # “multi:softmax” –set XGBoost to do multiclass classification
          #         using the softmax objective,
          #         you also need to set num_class(number of classes)
          # “multi:softprob” –same as softmax,
          #         but output a vector of ndata * nclass,
          #         which can be further reshaped to ndata, nclass matrix.
          #         The result contains predicted probability of each data
          #         point belonging to each class.
          # “rank:pairwise” –set XGBoost to do ranking task
          #         by minimizing the pairwise loss
          # “reg:gamma” –gamma regression with log-link.
          #         Output is a mean of gamma distribution.
          #         It might be useful,
          #         e.g., for modeling insurance claims severity,
          #         or for any outcome that might be gamma-distributed
          # “reg:tweedie” –Tweedie regression with log-link.
          #         It might be useful,
          #         e.g., for modeling total loss in insurance,
          #         or for any outcome that might be Tweedie-distributed.
          'eval_metric': 'rmse',
          # “rmse”: root mean square error
          # “mae”: mean absolute error
          # “logloss”: negative log-likelihood
          # “error”: Binary classification error rate.
          #         It is calculated as #(wrong cases)/#(all cases). For the predictions, the evaluation will regard the instances with prediction value larger than 0.5 as positive instances, and the others as negative instances.
          # “error@t”: a different than 0.5 binary classification
          #         threshold value could be specified by providing a
          #         numerical value through ‘t’.
          # “merror”: Multiclass classification error rate.
          #         It is calculated as #(wrong cases)/#(all cases).
          # “mlogloss”: Multiclass logloss
          # “auc”: Area under the curve for ranking evaluation.
          # “ndcg”: Normalized Discounted Cumulative Gain
          # “map”: Mean average precision
          # “ndcg@n”,”map@n”: n can be assigned
          #         as an integer to cut off the top positions
          #         in the lists for evaluation.
          # “ndcg-”,”map-”,”ndcg@n-”,”map@n-”: In XGBoost,
          #         NDCG and MAP will evaluate the score of a list
          #         without any positive samples as 1.
          #         By adding “-” in the evaluation metric
          #         XGBoost will evaluate these score as 0
          #         to be consistent under some conditions.
          #         training repeatedly
          'seed': SEED}

d_train = xgb.DMatrix(train_x, train_y)
d_valid = xgb.DMatrix(valid_X, valid_y)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

xgb.train(params=params,
          dtrain=d_train,
          num_boost_round=10,
          evals=watchlist,
          # List of items to be evaluated during training,
          # this allows user to watch performance on the validation set.
          obj=None,  # Customized objective function.
          feval=None,  # Customized evaluation function.
          maximize=False,  # Whether to maximize feval.
          early_stopping_rounds=None,
          evals_result=None,
          # This dictionary stores the evaluation results of all the items
          # in watchlist
          verbose_eval=True,
          xgb_model=None,
          # Xgb model to be loaded before training
          # (allows training continuation).
          callbacks=None,
          # List of callback functions
          # that are applied at end of each iteration.
          )

xgb.cv(params=params,
       dtrain=xgb.DMatrix(train_x, train_y),
       num_boost_round=10,
       nfold=5,
       stratified=False,
       folds=None,  # Sklearn KFolds or StratifiedKFolds instance.
       metrics=(),  # Evaluation metrics to be watched in CV (string).
       obj=None,  # Customized objective function.
       feval=None,  # Customized evaluation function.
       maximize=False,  # Whether to maximize feval.
       early_stopping_rounds=None,
       fpreproc=None,
       as_pandas=True,
       verbose_eval=None,
       show_stdv=True,
       seed=0,
       callbacks=None,  # List of callback functions
       # that are applied at end of each iteration.
       shuffle=True)

# sklearn cv (w/o early stopping?)
'''
brute force scan for all parameters, here are the tricks
usually max_depth is 6,7,8
learning rate is around 0.05, but small changes may make big diff
tuning min_child_weight subsample colsample_bytree can have
much fun of fighting against overfit
n_estimators is how many round of boosting
finally, ensemble xgboost with multiple seeds may reduce variance

'''
params = {'learning_rate': [0.05],
          'gamma': [0],
          'max_depth': [10],
          'min_child_weight': [1],
          # 'max_delta_step': 0,
          'subsample': [0.9],
          'colsample_bytree': [0.9],
          'colsample_bylevel': [0.9],
          'reg_lambda': [1],  # L2 regularization term on weights.
          'reg_alpha': [0],  # L1 regularization term on weights.
          'scale_pos_weight': [1]
          }
xgb_model = xgb.XGBClassifier(n_estimators=1000,
                              silent=True,
                              objective='binary:logistic',
                              nthread=1,
                              min_child_weight=1,
                              max_delta_step=0,
                              # base_score=0.5,
                              seed=SEED)
clf = GridSearchCV(xgb_model,
                   params,
                   cv=5,
                   n_jobs=3,
                   scoring='roc_auc',
                   verbose=0)
clf.fit(train_x, train_y)  # DMatrix can't be used

"""
## custom objective function and eval-metric function
## user define objective function,
# given prediction,
# return gradient and second order gradient
# this is log likelihood loss
"""


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess


"""
## user defined evaluation function,
# return a pair metric_name, result
# NOTE: when you do customized loss function,
# the default prediction value is margin
# this may make builtin evaluation metric not function properly
# for example, we are doing logistic loss,
# the prediction is score before logistic transformation
# the builtin evaluation error assumes input is after logistic transformation
# Take this in mind when you use the customization,
# and maybe you need write customized evaluation function
"""


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result.
    # The metric name must not contain a colon (:) or a space
    # since preds are margin(before logistic transformation, cutoff at 0)
    # see more detail example...
    # https://github.com/dmlc/xgboost/issues/1125#issuecomment-211711331
    return 'my-error', float(sum(labels != (preds > 0.0))) / len(labels)


# training with customized objective, we can also do step by step training
# simply look at xgboost.py's implementation of train
bst = xgb.train(params, dtrain, num_round, watchlist,
                obj=logregobj, feval=evalerror)

# scikit-learn ExtraTreesClassifier..................................
import gc
from time import time
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

pipe_ext = make_pipeline(
    ExtraTreesClassifier(
        random_state=SEED,
        n_jobs=CPU,
    )
)
param_grid_ext = {
    'extratreesclassifier__n_estimators': [1000],
    'extratreesclassifier__max_depth': [4, 6, 8],
    'extratreesclassifier__min_samples_split': [10],
    'extratreesclassifier__min_samples_leaf': [10],
    'extratreesclassifier__max_features': ['sqrt'],
    'extratreesclassifier__n_jobs': [CPU]
}
gridcv_ext = GridSearchCV(
    pipe_ext,
    param_grid=param_grid_ext,
    scoring='roc_auc',
    n_jobs=1,
    cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED)
)
st = time()
gridcv_ext.fit(train_x, train_y)
gc.collect()
etr_best_params = gridcv_ext.best_params_
message = '\nelapsed time : {:.1f} (m)\nBest AUC : {:.6f}'.format(
    (time() - st) / 60, gridcv_ext.best_score_)
print('\n' * 2,
      pd.DataFrame(gridcv_ext.cv_results_)[[
          'mean_test_score',
          'std_test_score',
          'param_extratreesclassifier__max_depth',
          'param_extratreesclassifier__n_estimators',
          'param_extratreesclassifier__min_samples_split',
          'param_extratreesclassifier__min_samples_leaf',
      ]].sort_values('mean_test_score', ascending=False))
print('\n', pd.Series(gridcv_ext.best_params_))
print(message)
# notifier(message)
# notifier(['\n' + i + ' : ' + str(etr_best_params[i]) for i in
#           etr_best_params.keys()])

# scikit-learn RandomForestClassifier..................................
import gc
from time import time
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

pipe_rf = make_pipeline(
    RandomForestClassifier(
        random_state=SEED,
        n_jobs=CPU,
    )
)
param_grid_rf = {
    'randomforestclassifier__n_estimators': [1000],
    'randomforestclassifier__max_depth': [4, 6, 8],
    'randomforestclassifier__min_samples_split': [10],
    'randomforestclassifier__min_samples_leaf': [10],
    'randomforestclassifier__max_features': ['sqrt'],
    'randomforestclassifier__class_weight': [None],
}
gridcv_rf = GridSearchCV(
    pipe_rf,
    param_grid=param_grid_rf,
    scoring='roc_auc',
    n_jobs=1,
    cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED)
)
st = time()
gridcv_rf.fit(train_x, train_y)
gc.collect()
rf_best_params = gridcv_rf.best_params_
message = '\nelapsed time : {:.1f} (m)\nBest AUC : {:.6f}'.format(
    (time() - st) / 60, gridcv_rf.best_score_)
print('\n' * 2,
      pd.DataFrame(gridcv_rf.cv_results_)[[
          'mean_test_score',
          'std_test_score',
          'param_randomforestclassifier__max_depth',
          'param_randomforestclassifier__n_estimators',
          'param_randomforestclassifier__min_samples_split',
          'param_randomforestclassifier__min_samples_leaf',
      ]].sort_values('mean_test_score', ascending=False))
print('\n', pd.Series(gridcv_rf.best_params_))
print(message)
# notifier(message)
# notifier(['\n' + i + ' : ' + str(rf_best_params[i]) for i in
#           rf_best_params.keys()])

# CatBoost Regressor.................................................
# https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_parameters-list-docpage/#python-reference_parameters-list

pool = cb.Pool(
    data=train_x,
    label=train_y,
    feature_names=boston.feature_names.tolist(),
)

params = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'learning_rate': 0.03,
    'random_seed': SEED,
    'l2_leaf_reg': 3,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.66,
    'sampling_frequency': 'PerTreeLevel',
    'depth': 2,
    'has_time': False,
    'rsm': 1.,  # colsample_bylevel
    'nan_mode': 'Min',  # 'Forbidden', 'Min', 'Max'
    # 'scale_pos_weight': 1.,
    'thread_count': CPU,
    'metric_period': 100,
    'verbose': 100,
}

scores = cb.cv(
    pool=pool,  # The input dataset to cross-validate.
    params=params,  # The list of parameters to start training with.
    iterations=999999,  # The maximum number of trees
    fold_count=5,
    inverted=False,  # invert train and test
    seed=SEED,  # the seed value for random permutation of the data.
    shuffle=True,  # Shuffle the dataset objects before splitting into folds.
    logging_level=None,  # 'Silent', 'Verbose', 'Info', 'Debug'
    stratified=False,  # Perform stratified sampling.
    as_pandas=True,  # Sets the type of return value to pandas.DataFrame.
    metric_period=1,  # Frequency of iter to calculate objectives and metrics.
    plot=False,  # only for jupyter notebook
    early_stopping_rounds=1000,
)
# 0.30726@7892
