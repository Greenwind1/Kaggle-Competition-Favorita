# -*- coding: utf-8 -*-

import gc
import os
import warnings
import psutil
import json
import pickle
import collections as cl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import umap
from sklearn.datasets import load_digits

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None
# dir(pd.options.display)
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')

digits = load_digits()

um = umap.UMAP(n_neighbors=5,
               min_dist=0.3,
               metric='correlation')
emb = um.fit_transform(digits.data)

fig, ax = plt.subplots(figsize=(7, 7))
for i in np.unique(digits.target):
    ax.plot(emb[digits.target == i, 0],
            emb[digits.target == i, 1],
            '.',
            markersize=5,
            color=plt.cm.Set1(i / 10.),
            alpha=0.7,
            label='{}'.format(i))
fig.legend()
fig.tight_layout()
fig.show()
