# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
#   Divide the training data into N (stratified by label) folds
# -------------------------------------------------------------------

import pickle
import psutil
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')

# -------------------------------------------------------------------
#   Environment Setting
# -------------------------------------------------------------------
SCRIPT_NAME = 'fold-01'
SEED = 2019
CPU = psutil.cpu_count()
FOLDS = 5
# FOLDS = 10

# -------------------------------------------------------------------
# Load Data
# -------------------------------------------------------------------
tr = pd.read_csv('./input/train_v2.csv')
te = pd.read_csv('./input/sample_submission_v2.csv')

# -------------------------------------------------------------------
# Extract Unique Labels
# -------------------------------------------------------------------
label_list = []
for tag_str in tr.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)

# -------------------------------------------------------------------
# Resolve into each Label
# -------------------------------------------------------------------
for label in label_list:
    tr[label] = tr['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
tr.head()
tr_y = tr.loc[:, 'haze':'blow_down']

# -------------------------------------------------------------------
# Create Folds
# -------------------------------------------------------------------
mskf = MultilabelStratifiedKFold(n_splits=FOLDS,
                                 shuffle=False,
                                 random_state=SEED)
for train_index, test_index in mskf.split(tr, tr_y.values):
    print("TRAIN:", train_index, "TEST:", test_index)
    print(tr_y.iloc[test_index].sum())

folds = list(mskf.split(tr, tr_y.values))
folds_fnames = [(tr['image_name'].values[f[0]],
                 tr['image_name'].values[f[1]]) for f in folds]

with open('./fold/{}_{}folds-{}seed.pkl'.format(SCRIPT_NAME, FOLDS, SEED),
          'wb') as f:
    pickle.dump(folds_fnames, f)

pd.concat(
    [tr[['image_name']], tr.loc[:, 'haze':'blow_down']], axis=1
).to_csv('./input/train_y.csv', index=False)
