# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import keras as kr
import keras.backend as K
from keras.utils import Sequence
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import (Input, Dense, BatchNormalization, Flatten,
                          Conv1D, Conv2D, GlobalAveragePooling1D,
                          GlobalAveragePooling2D, concatenate)
from keras.models import Model
from keras.optimizers import Adam
from keras.objectives import binary_crossentropy
from keras.metrics import categorical_crossentropy

from config import Config

config = Config()


# -------------------------------------------------------------------
#   Classification
# -------------------------------------------------------------------
def mnv2():
    mn = MobileNetV2(include_top=False, weights='imagenet')
    inp = Input(shape=(config.img_size, config.img_size, config.channel))
    mn_out = mn(inp)
    x = GlobalAveragePooling2D()(mn_out)
    x = Dense(1280, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(config.n_labels, activation='softmax')(x)
    model = Model(inputs=[inp], outputs=x)

    return model


def mnv2_1():
    mn = MobileNetV2(include_top=False, weights='imagenet')
    inp = Input(shape=(config.img_size, config.img_size, config.channel))
    x = BatchNormalization()(inp)
    mn_out = mn(x)
    x = GlobalAveragePooling2D()(mn_out)
    x = Dense(1280, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(config.n_labels, activation='softmax')(x)
    model = Model(inputs=[inp], outputs=x)

    return model


def mnv2_2():
    mn = MobileNetV2(include_top=False, weights='imagenet')
    inp = Input(shape=(config.img_size, config.img_size, config.channel))
    x = BatchNormalization()(inp)
    mn_out = mn(x)
    x = GlobalAveragePooling2D()(mn_out)
    x = Dense(1280, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(config.n_labels, activation='softmax')(x)
    model = Model(inputs=[inp], outputs=x)

    return model


def mnv2_3():
    mn = MobileNetV2(include_top=False, weights='imagenet')
    inp = Input(shape=(config.img_size, config.img_size, config.channel))
    x = BatchNormalization()(inp)
    mn_out = mn(x)
    x = GlobalAveragePooling2D()(mn_out)
    x = Dense(640, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(config.n_labels, activation='softmax')(x)
    model = Model(inputs=[inp], outputs=x)

    return model


# -------------------------------------------------------------------
#   Regression
# -------------------------------------------------------------------
def mnv2_10():
    mn = MobileNetV2(include_top=False, weights='imagenet')
    inp = Input(shape=(config.img_size, config.img_size, config.channel))
    x = BatchNormalization()(inp)
    mn_out = mn(x)
    x = GlobalAveragePooling2D()(mn_out)
    x = Dense(1280, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1)(x)
    model = Model(inputs=[inp], outputs=x)

    return model
