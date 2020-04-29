# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import keras as ks
import keras.backend as K
from keras.utils import Sequence
# from keras.applications.mobilenetv2 import MobileNetV2
from classification_models.resnet import ResNet34
from keras_applications.resnet50 import ResNet50
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
#   Regression
# -------------------------------------------------------------------
def rn34_01():
    base = ResNet34(include_top=False,
                    weights='imagenet',
                    backend=ks.backend,
                    layers=ks.layers,
                    utils=ks.utils,
                    models=ks.models)
    base.layers.pop(0)
    inp = Input(shape=(config.img_size, config.img_size, config.channel))
    x = BatchNormalization()(inp)
    base_out = base(x)
    x = GlobalAveragePooling2D()(base_out)
    x = Dense(1280, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1)(x)
    model = Model(inputs=[inp], outputs=x)

    return model


def rn50_10():
    base = ResNet50(include_top=False,
                    weights='imagenet',
                    backend=ks.backend,
                    layers=ks.layers,
                    utils=ks.utils,
                    models=ks.models)
    base.layers.pop(0)
    inp = Input(shape=(config.img_size, config.img_size, config.channel))
    x = BatchNormalization()(inp)
    base_out = base(x)
    x = GlobalAveragePooling2D()(base_out)
    x = Dense(1280, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1)(x)
    model = Model(inputs=[inp], outputs=x)

    return model
