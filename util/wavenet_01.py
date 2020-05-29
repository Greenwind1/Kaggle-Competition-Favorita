# -*- coding: utf-8 -*-
"""

coded by HoxoMaxwell
 
"""
import os, gc, sys, warnings
import psutil
import json
import pickle
import collections as cl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from datetime import date
from keras.models import Model
from keras.layers import *

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None
# dir(pd.options.display)
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')
sys.path.append('')

# -------------------------------------------------------------------
#   Set Env
# -------------------------------------------------------------------
TIMESTEPS = 28 * 13
PRED_DAYS = 28
CAT_INPUT_N = 6
WAVENET_LATENT_DIM = 32

# Unique Size of Categorical Feature
CAT_EMBED_1 = 33
CAT_EMBED_2 = 54
CAT_EMBED_3 = 17
CAT_EMBED_4 = 5
CAT_EMBED_5 = 5

# -------------------------------------------------------------------
#   WaveNet Model
# -------------------------------------------------------------------
# -----------------------------------------------
#   Functions
# -----------------------------------------------
# slice sequence into past one
encode_slice = Lambda(lambda x: x[:, :TIMESTEPS, :])

# slice sequence into future one
decode_slice = Lambda(lambda x: x[:, TIMESTEPS:, :])

# -----------------------------------------------
#   Sequential Input
# -----------------------------------------------
y_in = Input(shape=(TIMESTEPS, 1))
# is0_in = Input(shape=(TIMESTEPS, 1))
price_in = Input(shape=(TIMESTEPS + 16, 1))
item_mean_in = Input(shape=(TIMESTEPS, 1))
# store_mean_in = Input(shape=(TIMESTEPS, 1))
# weekday_in = Input(shape=(TIMESTEPS + 16,), dtype='uint8')
# weekday_embed_encode = Embedding(7, 4, input_length=TIMESTEPS + 16)(weekday_in)
# dom_in = Input(shape=(TIMESTEPS + 16,), dtype='uint8')
# dom_embed_encode = Embedding(31, 4, input_length=TIMESTEPS + 16)(dom_in)

# -----------------------------------------------
#   Categorical Input
# -----------------------------------------------
cat_in = Input(shape=(CAT_INPUT_N,))
item_id = Lambda(lambda x: x[:, 0, None])(cat_in)
dept_id = Lambda(lambda x: x[:, 1, None])(cat_in)
cat_id = Lambda(lambda x: x[:, 2, None])(cat_in)
store_id = Lambda(lambda x: x[:, 3, None])(cat_in)
state_id = Lambda(lambda x: x[:, 4, None])(cat_in)

item_embed = Embedding(CAT_EMBED_1, 8, input_length=1)(item_id)
dept_embed = Embedding(CAT_EMBED_2, 8, input_length=1)(dept_id)
cat_embed = Embedding(CAT_EMBED_3, 8, input_length=1)(cat_id)
store_embed = Embedding(CAT_EMBED_4, 8, input_length=1)(store_id)
state_embed = Embedding(CAT_EMBED_5, 3, input_length=1)(state_id)

# -----------------------------------------------
#   Future Price Input
# -----------------------------------------------
price_pred = decode_slice(price_in)

# -----------------------------------------------
#   Auxiliary Input
# -----------------------------------------------


# -----------------------------------------------
#   WaveNet for Sequential Input
# -----------------------------------------------
wavenet_in = concatenate([
    y_in,
    encode_slice(price_in),
    item_mean_in
], axis=2)

c1 = Conv1D(filters=WAVENET_LATENT_DIM, kernel_size=2,
            dilation_rate=1, padding='causal', activation='relu')(wavenet_in)
c2 = Conv1D(filters=WAVENET_LATENT_DIM, kernel_size=2,
            dilation_rate=2, padding='causal', activation='relu')(c1)
c2 = Conv1D(filters=WAVENET_LATENT_DIM, kernel_size=2,
            dilation_rate=4, padding='causal', activation='relu')(c2)
c2 = Conv1D(filters=WAVENET_LATENT_DIM, kernel_size=2,
            dilation_rate=8, padding='causal', activation='relu')(c2)
# c2 = Conv1D(filters=WAVENET_LATENT_DIM, kernel_size=2,
#             dilation_rate=16, padding='causal', activation='relu')(c2)
wavenet_out = concatenate([c1, c2])
conv_out = Conv1D(8, 1, activation='relu')(wavenet_out)
conv_out = Dropout(0.25)(conv_out)
conv_out = Flatten()(conv_out)

dnn_out = Dense(512, activation='relu')(Flatten()(y_in))
dnn_out = Dense(256, activation='relu')(dnn_out)
# dnn_out = BatchNormalization()(dnn_out)
dnn_out = Dropout(0.25)(dnn_out)

# -----------------------------------------------
#   Concat
# -----------------------------------------------
x = concatenate([
    conv_out,
    dnn_out,
    Flatten()(price_pred),
    Flatten()(item_embed),
    Flatten()(dept_embed),
    Flatten()(cat_embed),
    Flatten()(store_embed),
    Flatten()(state_embed),
])

# -----------------------------------------------
#   Top FC Layers
# -----------------------------------------------
# x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.25)(x)
# x = Dense(256, activation='relu')(x)
# x = BatchNormalization()(x)
# x = concatenate([x, seq_in])
output = Dense(PRED_DAYS, activation='relu')(x)

# -----------------------------------------------
#   Build
# -----------------------------------------------
model = Model(
    [
        y_in,
        # is0_in,
        price_in,
        item_mean_in,
        # store_mean_in,
        # quarterAgo_in,
        # weekday_in,
        # dom_in,
        cat_in,
    ],
    output
)
