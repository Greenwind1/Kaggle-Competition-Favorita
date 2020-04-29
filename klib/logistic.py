# -*- coding: utf-8 -*-

# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

import pandas as pd
import numpy as np
import gc

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import l1_min_c
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

"""
ENVIROMENT VARIABLES
"""
CODE_N = 'logit'

CPU = psutil.cpu_count() - 1
FI = True
# FI = False
# F_SAVE = True
F_SAVE = False
# F_DROP = True
F_DROP = False
DEBUG = True
# DEBUG = False

SEED = 71
SEED_LIST = [SEED, 72, 73, 74, 75, 76]

X = train.drop(['id', 'target'], axis=1)
X_test = test.copy()
X = X.fillna(-1)
X_test = X_test.fillna(-1)

y = train['target'].values

# imputation
X = X.replace(-1, np.NaN)
X_test = X_test.replace(-1, np.NaN)

imp = Imputer(missing_values='NaN', strategy='most_frequent')
for i in X.columns[X.isnull().sum() > 0]:
    X[i] = imp.fit_transform(X[i].values.reshape(-1, 1))
for i in X_test.columns[X_test.isnull().sum() > 0]:
    X_test[i] = imp.fit_transform(X_test[i].values.reshape(-1, 1))

# categorization
for c in selected_col:
    if ('_cat' in c) and ('_te' not in c) and ('_ohe_' not in c):
        X[c] = X[c].apply(str)
        X_test[c] = X_test[c].apply(str)

X = pd.get_dummies(X)
X = X.values
X_test = pd.get_dummies(X_test)
gc.collect()

# ---------------------------------------------------------------------
# Grid Search

# Scaling
ss = StandardScaler()
mm = MinMaxScaler()
ss.fit(pd.concat([train_x, test_x], axis=0))
train_x_s = ss.transform(train_x)
test_x_s = ss.transform(test_x)
mm.fit(pd.concat([train_x, test_x], axis=0))
train_x_m = mm.transform(train_x)
test_x_m = mm.transform(test_x)

cs = l1_min_c(X, y, loss='log')  # lower limit of 'c' in L1 regression
param_grid = {'penalty': ['l1'],
              'C': [0.1, 0.2]}
grid_cv_logit = GridSearchCV(
    LogisticRegression(
        solver='saga',  # L1 : sag, L2 :saga, both algo need Scaling
        random_state=SEED,
        n_jobs=1),
    param_grid=param_grid,
    scoring='roc_auc',
    n_jobs=CPU,
    cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED),
    verbose=1,
)

grid_cv_logit.fit(train_x_s, train_y.values.reshape(-1, ))
# grid_cv_logit.fit(train_x_m, train_y.values.reshape(-1, ))

logit_best_params = grid_cv_logit.best_params_
logit_cv_result = pd.DataFrame(grid_cv_logit.cv_results_)
logit_cv_rank = logit_cv_result.sort_values('rank_test_score')[
    ['param_' + k for k in param_grid.keys()]]
logger.info('\n{}'.format(logit_cv_rank))
# notifier(CODE_N + '\n{}'.format(logit_cv_rank))

m = '{} \nAUC : {:.6f}'.format(logit_best_params, grid_cv_logit.best_score_)
logger.info('\n{}'.format(m))
notifier(CODE_N + '\n' + m)
gc.collect()
# ---------------------------------------------------------------------


seed = 0
kfold = 5
skf = StratifiedKFold(n_splits=kfold, shuffle=False, random_state=seed)
pred_df = test['id'].to_frame()
s_train = train['id'].to_frame()
s_train['target'] = 0.
logit_cv_score = []
sub = test['id'].to_frame()
sub['target'] = 0

for i, (train_index, valid_index) in enumerate(skf.split(X, y)):
    print('-' * 80)
    print(' logit kfold: {}  of  {} : '.format(i + 1, kfold))
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    logit_model = LogisticRegression(penalty='l1',
                                     C=0.1,
                                     solver='liblinear',
                                     class_weight='balanced',
                                     random_state=seed,
                                     n_jobs=2)
    logit_model.fit(scaler.fit_transform(X_train), y_train)
    pred_test = logit_model.predict_proba(scaler.fit_transform(X_test))[:,
                1] / kfold
    sub['target'] += pred_test
    pred_valid = pd.DataFrame(
        logit_model.predict_proba(scaler.fit_transform(X_valid)),
        index=valid_index)
    s_train.loc[valid_index, 'target'] = pred_valid.values[:, 1]
    pred_df = pd.concat((pred_df, pd.DataFrame(pred_test * kfold)), axis=1)
    logit_cv_score.append(roc_auc_score(y_valid,
                                        pred_valid.values[:, 1]) * 2 - 1)
    print(roc_auc_score(y_valid, pred_valid.values[:, 1]) * 2 - 1)

gc.collect()
logit_local_cv = round(sum(logit_cv_score) / kfold, 6)
print([round(i, 6) for i in logit_cv_score], logit_local_cv)
# 'C': 0.02 [0.28188, 0.278632, 0.276523, 0.276754, 0.276421] 0.278042
# 'C': 0.2 [0.282874, 0.279513, 0.277454, 0.277088, 0.277666] 0.278919
