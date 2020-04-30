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

from util.Utils import load_data, save_unstack

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None
# dir(pd.options.display)
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')
sys.path.append('')

df_name = './input/unstack_train.f'
promo_name = './input/unstack_promo.f'

df, promo = load_data()

df.columns = df.columns.astype('str')
promo.columns = promo.columns.astype('str')

df.loc[:, '2016-12-25'] = 0
df.sort_index(axis=1, inplace=True)

promo.loc[:, '2016-12-25'] = False
promo.sort_index(axis=1, inplace=True)

df.reset_index().to_feather(df_name)
promo.reset_index().to_feather(promo_name)
