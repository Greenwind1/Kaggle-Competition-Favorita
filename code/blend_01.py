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

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None
# dir(pd.options.display)
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')
sys.path.append('')

pred_l = [
    # LGB
    pd.read_csv('./sub/lgb/lgb_seed2020.csv'),
    pd.read_csv('./sub/lgb/lgb_seed2021.csv'),
    pd.read_csv('./sub/lgb/lgb_seed2022.csv'),
    pd.read_csv('./sub/lgb/lgb_seed2023.csv'),
    pd.read_csv('./sub/lgb/lgb_seed2024.csv'),
    # CNN
    pd.read_csv('./sub/dnn/cnn_epoch5.csv'),
    pd.read_csv('./sub/dnn/cnn_epoch10.csv'),
    pd.read_csv('./sub/dnn/cnn_epoch15.csv'),
    pd.read_csv('./sub/dnn/cnn_epoch20.csv'),
    pd.read_csv('./sub/dnn/cnn_epoch25.csv'),
]

pred_b = pred_l[0][['id']].copy()
pred_b['unit_sales'] = 0

for p in pred_l:
    pred_b['unit_sales'] += p['unit_sales'] / len(pred_l)

pred_b.to_csv(f'./sub/blend/blend_01-{len(pred_l)}.csv',
              index=False)