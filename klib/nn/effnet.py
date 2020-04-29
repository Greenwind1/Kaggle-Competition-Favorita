# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import keras as ks
import keras.backend as K
from keras.utils import Sequence
from efficientnet import (EfficientNetB0,
                          EfficientNetB1,
                          EfficientNetB2,
                          EfficientNetB3,
                          EfficientNetB4,
                          EfficientNetB5)
from keras.layers import (Input, Dense, BatchNormalization, Flatten,
                          Conv1D, Conv2D, GlobalAveragePooling1D,
                          GlobalAveragePooling2D, concatenate, Lambda,
                          Activation, Dropout)
from keras.models import Model
from keras.optimizers import Adam
from keras.objectives import binary_crossentropy
from keras.metrics import categorical_crossentropy

from tensorflow.contrib.distributions import percentile

from config import Config

config = Config()


# -------------------------------------------------------------------
#   Helper Functions
# -------------------------------------------------------------------
def stat_tensor(tensors):
    out1 = K.mean(tensors, axis=(1, 2))
    out2 = K.std(tensors, axis=(1, 2))
    out3_5 = percentile(tensors, q=5., axis=(1, 2))
    out3_15 = percentile(tensors, q=15., axis=(1, 2))
    out3_35 = percentile(tensors, q=35., axis=(1, 2))
    out3_50 = percentile(tensors, q=50., axis=(1, 2))
    out3_65 = percentile(tensors, q=65., axis=(1, 2))
    out3_85 = percentile(tensors, q=85., axis=(1, 2))
    out3_95 = percentile(tensors, q=95., axis=(1, 2))
    return K.concatenate(
        [out1, out2,
         out3_5, out3_15, out3_35, out3_50, out3_65, out3_85, out3_95],
        axis=1
    )


# -------------------------------------------------------------------
#   Regression
#   IMG SIZE (B0: 224, B1: 240, B2: 260, B3: 300, B4: 380, B5: 456)
# -------------------------------------------------------------------

# def eff_b2_01():
#     # 260
#     K.clear_session()
#     base = EfficientNetB2(include_top=False, weights='imagenet')
#     base.layers.pop(0)
#     inp = Input(shape=(config.img_size, config.img_size, config.channel))
#     # x = BatchNormalization()(inp)
#     base_out = base(inp)
#     x = GlobalAveragePooling2D()(base_out)
#     x = Dense(1280)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Dense(128)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Dense(1)(x)
#     model = Model(inputs=inp, outputs=x)
#
#     return model


# def eff_b2_stat_01():
#     # 260, stat
#     base = EfficientNetB2(include_top=False, weights='imagenet')
#     base.layers.pop(0)
#     inp = Input(shape=(config.img_size, config.img_size, config.channel))
#     x = BatchNormalization()(inp)
#     base_out = base(x)
#     stat_out = Lambda(function=stat_tensor, name='stat-lambda')(x)
#     stat_out = BatchNormalization(name='stat-bn')(stat_out)
#
#     base_out = GlobalAveragePooling2D()(base_out)
#     out = concatenate([base_out, stat_out], axis=1)
#     out = Dense(1280, activation='relu')(out)
#     out = BatchNormalization()(out)
#     out = Dense(128, activation='relu')(out)
#     out = BatchNormalization()(out)
#     out = Dense(1)(out)
#     model = Model(inputs=[inp], outputs=out)
#
#     return model


# -------------------------------------------------------------------
#   Classification
# -------------------------------------------------------------------

# Multi-Label
def eff_b0_00():
    # 224
    base = EfficientNetB0(
        include_top=False,
        input_shape=(config.img_size, config.img_size, config.channel),
        weights='imagenet'
    )

    # Using BN to normalize images
    # base.layers.pop(0)
    # inp = Input(shape=(config.img_size, config.img_size, config.channel))
    # x = BatchNormalization()(inp)
    # base_out = base(inp)

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(1280)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(config.n_labels, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=x)

    return model


def eff_b0_01():
    # 224
    base = EfficientNetB0(
        include_top=False,
        input_shape=(config.img_size, config.img_size, config.channel),
        weights=None
    )

    # Using BN to normalize images
    # base.layers.pop(0)
    # inp = Input(shape=(config.img_size, config.img_size, config.channel))
    # x = BatchNormalization()(inp)
    # base_out = base(inp)

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(1280)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(config.n_labels, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=x)

    return model

# # Multi-Label
# def eff_b2_01():
#     # 260
#     base = EfficientNetB2(include_top=False, weights='imagenet')
#     base.layers.pop(0)
#     inp = Input(shape=(config.img_size, config.img_size, config.channel))
#     # x = BatchNormalization()(inp)
#     base_out = base(inp)
#     x = GlobalAveragePooling2D()(base_out)
#     x = Dense(1280)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Dense(128)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Dense(config.n_labels, activation='sigmoid')(x)
#     model = Model(inputs=inp, outputs=x)
#
#     return model

# def eff_b3_20():
#     # 300
#     base = EfficientNetB3(include_top=False, weights='imagenet')
#     base.layers.pop(0)
#     inp = Input(shape=(config.img_size, config.img_size, config.channel))
#     x = BatchNormalization()(inp)
#     base_out = base(x)
#     x = GlobalAveragePooling2D()(base_out)
#     x = Dense(1280, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dense(128, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dense(5, activation='softmax')(x)
#     model = Model(inputs=[inp], outputs=x)
#
#     return model
#
#
# def eff_b3_31():
#     # 300
#     # Rank-consistent Ordinal Regression
#     base = EfficientNetB3(include_top=False, weights='imagenet')
#     base.layers.pop(0)
#     inp = Input(shape=(config.img_size, config.img_size, config.channel))
#     x = BatchNormalization()(inp)
#     base_out = base(x)
#     x = GlobalAveragePooling2D()(base_out)
#     x = Dense(1280, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dense(192, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dense(4, activation='sigmoid')(x)
#     model = Model(inputs=[inp], outputs=x)
#
#     return model
