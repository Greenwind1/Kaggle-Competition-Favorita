# -*- coding: utf-8 -*-
"""

coded by HoxoMaxwell
 
"""
import os
import gc
import sys
import warnings
import psutil
import json
import pickle
import collections as cl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from glob import glob

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None
# dir(pd.options.display)
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')
sys.path.append('')

csv_l = np.sort(glob('./sub/blend/ensemble_cnn_0*-5.csv')).tolist()
csv_l += np.sort(glob('./sub/blend/blend_lgb_02-10.csv')).tolist()
csv_l += np.sort(glob('./sub/blend/blend_lgb_*-20.csv')).tolist()
print(csv_l)
pred_l = [pd.read_csv(c) for c in csv_l]

w_l = [
    1, 1, 1, 1, 1, 1, 2, 2
]

print(f'len(csv_l) = {len(csv_l)} \nlen(w_l) = {len(w_l)}')

pred_b = pred_l[0][['id']].copy()
pred_b['unit_sales'] = 0

for i, p in enumerate(pred_l):
    pred_b['unit_sales'] += p['unit_sales'] * w_l[i] / np.sum(w_l)

pred_b.to_csv(f'./sub/blend/blend_05-{len(pred_l)}_{tuple(w_l)}.csv',
              index=False)
