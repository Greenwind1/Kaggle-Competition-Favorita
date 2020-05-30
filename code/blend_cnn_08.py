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

NAME = '08'

csv_l = glob(f'./sub/dnn/cnn_epoch-*_epoch*.csv')
csv_l += glob(f'./sub/blend/*cnn_02-5.csv')
print(f'{csv_l}')
pred_l = [pd.read_csv(c) for c in csv_l]

pred_b = pred_l[0][['id']].copy()
pred_b['unit_sales'] = 0

for p in pred_l:
    pred_b['unit_sales'] += p['unit_sales'] / len(pred_l)

pred_b.to_csv(f'./sub/blend/ensemble_cnn_{NAME}-{len(pred_l)}.csv',
              index=False)
