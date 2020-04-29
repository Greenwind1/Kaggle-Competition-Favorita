# -*- coding: utf-8 -*-

class Stacking(object):
    import pandas as pd
    import numpy as np
    import gc
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score

    def __init__(self, n_splits, shuffle,
                 stacker, cv_stack, base_learners, seed):

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.stacker = stacker
        self.cv_stack = cv_stack
        self.base_learners = base_learners
        self.seed = seed

    def fit_predict(self, train_X, train_y, test):
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        test = np.array(test)

        folds = list(StratifiedKFold(n_splits=self.n_splits,
                                     shuffle=self.shuffle,
                                     random_state=self.seed).split(train_X,
                                                                   train_y))

        s_train = np.zeros((train_X.shape[0], len(self.base_learners)))
        s_test = np.zeros((test.shape[0], len(self.base_learners)))

        # Iterate each model and make meta features
        for i, clf in enumerate(self.base_learners):

            s_test_i = np.zeros((test.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = train_X[train_idx]
                y_train = train_y[train_idx]
                X_holdout = train_X[test_idx]
                # y_holdout = y[test_idx]

                print('-' * 80)
                print("Fit %s fold %d" % (str(clf).split('(')[0], j))
                clf.fit(X_train, y_train)
                # cross_score = cross_val_score(clf, X_train, y_train, cv=3,
                # scoring='roc_auc')
                # print("    cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:, 1]

                s_train[test_idx, i] = y_pred
                s_test_i[:, j] = clf.predict_proba(test)[:, 1]
            s_test[:, i] = s_test_i.mean(axis=1)

        results = cross_val_score(self.stacker,
                                  s_train, train_y,
                                  cv=self.cv_stack,
                                  scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        # fit stacker model
        self.stacker.fit(s_train, train_y)
        res = self.stacker.predict_proba(s_test)[:, 1]
        return res


if __name__ == '__main__':
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClaspipsifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from rgf.sklearn import RGFClassifier

    lgb_model = LGBMClassifier(**lgb_params)
    # key word arg (xxx = 10 or dictionary)

    xgb_model = XGBClassifier(**xgb_params)
    log_model = LogisticRegression()

    stack = Stacking(n_splits=5,
                     stacker=log_model,
                     base_learners=(lgb_model, xgb_model))

    y_pred = stack.fit_predict(train, target_train, test)
