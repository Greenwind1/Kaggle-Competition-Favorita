# -*- coding: utf-8 -*-

import gc
import os
import warnings
import psutil
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import cross_val_score
from fancyimpute import KNN, NuclearNormMinimization
from fancyimpute import SoftImpute, IterativeImputer, BiScaler

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

CPU = psutil.cpu_count() - 1
SEED = 71

# Note: Each of these 10 feature variables have been mean centered
# and scaled by the standard deviation times n_samples
dia = load_diabetes()
train_x = dia.data.copy()
train_y = dia.target.copy()
f_name = dia.feature_names


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


rmse_scorer = make_scorer(rmse,
                          greater_is_better=False,
                          needs_proba=False)

lrcv = RidgeCV(
    scoring=rmse_scorer,
    store_cv_values=True,
)
lrcv.fit(dia.data, train_y)
ridge = Ridge(alpha=lrcv.alpha_, random_state=SEED)
cvs = cross_val_score(ridge,
                      dia.data, train_y,
                      scoring=rmse_scorer,
                      cv=8,
                      )
print('CV RMSE mean : {:.2f}'.format(-cvs.mean()))

for i in range(5):
    print('\n{} th iteration'.format(i + 1))
    np.random.seed(SEED + i)

    train_x = dia.data.copy()
    train_y = dia.target.copy()
    mask = np.random.choice(len(dia.data), 50)
    masked_x = dia.data[mask, 2]
    train_x[mask, 2] = np.NaN

    # Model each feature with missing values as a function of other features,
    # and use that estimate for imputation.
    train_x_ii = IterativeImputer(
        random_state=SEED
    ).fit_transform(train_x)
    masked_ii = train_x_ii[mask, 2]

    # Use 3 nearest rows which have a feature to fill
    # in each row's missing features
    train_x_knn = KNN(k=3, verbose=False).fit_transform(train_x)
    masked_knn = train_x_knn[mask, 2]

    # matrix completion using convex optimization to find low-rank solution
    # that still matches observed values.
    # Slow!
    # train_x_nnm = NuclearNormMinimization().fit_transform(train_x)
    # imp_nnm = train_x_nnm[train_x.isnull().values]

    # Instead of solving the nuclear norm objective directly, instead
    # induce sparsity using singular value thresholding
    train_x_normalized = BiScaler(verbose=False).fit_transform(train_x)
    train_x_softimpute = SoftImpute(verbose=False).fit_transform(
        train_x_normalized)
    masked_soft = train_x_softimpute[mask, 2]

    # print mean squared error for the four imputation methods above
    ii_mse = ((masked_ii - masked_x) ** 2).mean()
    knn_mse = ((masked_knn - masked_x) ** 2).mean()
    soft_mse = ((masked_soft - masked_x) ** 2).mean()

    lrcv.fit(train_x_ii, train_y)
    print("Iterative Imputer\nImputed MSE : {:5f}".format(ii_mse))
    print('Ridge alpha : {}'.format(lrcv.alpha_))
    ridge = Ridge(alpha=lrcv.alpha_, random_state=SEED + i)
    cvs = cross_val_score(ridge,
                          train_x_ii, train_y,
                          scoring=rmse_scorer,
                          cv=8,
                          )
    print('CV RMSE mean : {:.2f}\n'.format(-cvs.mean()))

    lrcv.fit(train_x_knn, train_y)
    print("knnImpute\nImputed MSE : {:5f}".format(knn_mse))
    print('Ridge alpha : {}'.format(lrcv.alpha_))
    ridge = Ridge(alpha=lrcv.alpha_, random_state=SEED + i)
    cvs = cross_val_score(ridge,
                          train_x_knn, train_y,
                          scoring=rmse_scorer,
                          cv=8,
                          )
    print('CV RMSE mean : {:.2f}\n'.format(-cvs.mean()))

    lrcv.fit(train_x_softimpute, train_y)
    print("SoftImpute\nImputed MSE : {:5f}".format(soft_mse))
    print('Ridge alpha : {}'.format(lrcv.alpha_))
    ridge = Ridge(alpha=lrcv.alpha_, random_state=SEED + i)
    cvs = cross_val_score(ridge,
                          train_x_softimpute, train_y,
                          scoring=rmse_scorer,
                          cv=8,
                          )
    print('CV RMSE mean : {:.2f}\n'.format(-cvs.mean()))
