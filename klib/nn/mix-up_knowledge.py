# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os

from klib.rank_gauss import GaussRankScaler

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
from sklearn.metrics import roc_auc_score

from keras.layers import Activation, Dropout, Flatten, Dense, \
    GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam
from keras import backend as K
import keras
from keras.models import Model
from keras import regularizers
import tensorflow as tf
from keras.losses import binary_crossentropy
import gc
import scipy.special
from tqdm import *
from scipy.stats import norm, rankdata

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, \
    EarlyStopping, ReduceLROnPlateau

SEED = 2019
BATCH_SIZE = 1024

train = pd.read_feather('./input/train.f')
test = pd.read_feather('./input/test.f')
# Load Knowledge Distillation CSV (OOF Prediction)
train_knowledge = pd.read_csv(
    './sub/LGBM/lgb_3-2_20_SUB-SEEDS-AVG-0.900303.zip')

y = train['target']
y_knowledge = train_knowledge['TARGET_SEED_AVG']
id_code_train = train['ID_code']
id_code_test = test['ID_code']
features = [c for c in train.columns if c not in ['ID_code', 'target']]

SPLIT = len(train)
train = pd.concat([train.loc[:, features], test.loc[:, features]], axis=0)
del test
gc.collect()

scaler = GaussRankScaler()
sc = StandardScaler()

for feat in tqdm(features):
    train[feat] = scaler.fit_transform(train[feat])
    # train[feat] = sc.fit_transform(train[feat].values.reshape(-1, 1))
    train[feat + '_r'] = rankdata(train[feat]).astype('float32')
    train[feat + '_n'] = norm.cdf(train[feat]).astype('float32')

num_features = train.shape[1]

feats = [c for c in train.columns if
         c not in (['ID_code', 'target'] + features)]
for feat in tqdm(feats):
    train[feat] = sc.fit_transform(train[feat].values.reshape(-1, 1))

test = train[SPLIT:].values
train = train[:SPLIT].values
print(train.shape)

train = np.reshape(train, (-1, num_features, 1))
test = np.reshape(test, (-1, num_features, 1))

x_train, x_valid, y_train, y_valid, y_knowledge_train, y_knowledge_valid = \
    train_test_split(
        train, y, y_knowledge, stratify=y, test_size=0.2, random_state=SEED
    )

act_f = keras.layers.advanced_activations.LeakyReLU(alpha=.001)


# -------------------------------------------------------------------
# 1.
# SIMPLE NN_MODEL
def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    x = Dense(16, activation=act_f)(input_tensor)
    x = Flatten()(x)
    out_put = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, out_put)

    return model


# WRAPPER FUNCTION
# https://gangango.com/2018/09/24/post-225/
def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


# LOSS FUNCTION
# FOCAL LOSS
# https://arxiv.org/abs/1708.02002
gamma = 2.0
alpha = .25
epsilon = K.epsilon()


def focal_loss(y_true, y_pred):
    """
    pt_1 : P(t=1) * 1
    ce_1 : cross entropy
    fl_1 : focal loss ( = alpha * ( 1 - p )^gamma * ce )
    """
    pt_1 = y_pred * y_true
    pt_1 = K.clip(pt_1, epsilon, 1 - epsilon)
    ce_1 = -K.log(pt_1)
    fl_1 = alpha * K.pow(1 - pt_1, gamma) * ce_1

    pt_0 = (1 - y_pred) * (1 - y_true)
    pt_0 = K.clip(pt_0, epsilon, 1 - epsilon)
    ce_0 = -K.log(pt_0)
    fl_0 = (1 - alpha) * K.pow(1 - pt_0, gamma) * ce_0

    loss = K.sum(fl_1, axis=1) + K.sum(fl_0, axis=1)
    return loss


## MIXUP
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    sample_size = x.shape[0]
    index_array = np.arange(sample_size)
    np.random.shuffle(index_array)

    mixed_x = lam * x + (1 - lam) * x[index_array]
    mixed_y = (lam * y) + ((1 - lam) * y[index_array])

    return mixed_x, mixed_y


## BATCHES
def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in
            range(0, nb_batch)]


def batch_generator(x, y, batch_size=128, shuffle=True, mixup=False):
    y = np.array(y)
    sample_size = x.shape[0]
    index_array = np.arange(sample_size)

    while True:
        if shuffle:
            np.random.shuffle(index_array)

        batches = make_batches(sample_size, batch_size)

        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            x_batch = x[batch_ids]
            y_batch = y[batch_ids]

            if mixup:
                x_batch, y_batch = mixup_data(x_batch, y_batch, alpha=1.0)

            yield x_batch, y_batch


# SIMPLE DNN

model = create_model((num_features, 1), 1)
model.compile(loss='binary_crossentropy', optimizer='adam')
model.summary()

# MODEL SAVE
checkpoint = ModelCheckpoint(
    './model/feed_forward_model.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=True)

# LEARNING RATE SCHEDULER
reduceLROnPlat = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    verbose=1,
    mode='min',
    epsilon=0.0001)

# EARLY STOPPING
early = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=9)

# BATCH_GENERATOR
tr_gen = batch_generator(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    mixup=True)

# TRAIN AND PREDICT
history = model.fit_generator(
    tr_gen,
    steps_per_epoch=np.ceil(float(len(x_train)) / float(BATCH_SIZE)),
    epochs=20,
    verbose=1,
    callbacks=[checkpoint, reduceLROnPlat, early],
    validation_data=(x_valid, y_valid)
)

model.load_weights('./model/feed_forward_model.h5')
prediction = model.predict(x_valid, batch_size=512, verbose=1)
roc_auc_score(y_valid, prediction)

# -------------------------------------------------------------------
# 2.
# KNOWLEDGE DISTILLATION 1
y_train = np.vstack((y_train, y_knowledge_train)).T
y_valid = np.vstack((y_valid, y_knowledge_valid)).T

print(y_train.shape)
print(y_train[0])


def knowledge_distillation_bce(y_true, y_pred, beta=0.1):
    # Extract the groundtruth from dataset and
    # the prediction from teacher model
    y_true, y_pred_teacher = y_true[:, :1], y_true[:, 1:]

    # Extract the prediction from student model
    y_pred, y_pred_stu = y_pred[:, :1], y_pred[:, 1:]

    loss = beta * binary_crossentropy(y_true, y_pred) + (
            1 - beta) * binary_crossentropy(y_pred_teacher, y_pred_stu)

    return loss


def auc_2(y_true, y_pred):
    y_true = y_true[:, :1]
    y_pred = y_pred[:, :1]
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


def auc_3(y_true, y_pred):
    y_true = y_true[:, :1]
    y_pred = y_pred[:, 1:]
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


model = create_model((NUM_FEATURES, 1), 2)
model.compile(loss=knowledge_distillation_bce,
              optimizer='adam',
              metrics=[auc_2])

checkpoint = ModelCheckpoint(
    './model/student_model_be.h5',
    monitor='val_auc_2',
    verbose=0,
    save_best_only=True,
    mode='max',
    save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(
    monitor='val_auc_2', factor=0.5, patience=4,
    verbose=1, mode='max', epsilon=0.0001)

early = EarlyStopping(monitor="val_auc_2",
                      mode="max",
                      patience=9)

history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, reduceLROnPlat, early],
    validation_data=(x_valid, y_valid))


# -------------------------------------------------------------------
## KNOWLEDGE DISTILLATION 2 (WITH FOCAL LOSS)
def knowledge_distillation_fl(y_true, y_pred, beta=0.1):
    # Extract the groundtruth from dataset and
    # the prediction from teacher model
    y_true, y_pred_teacher = y_true[:, :1], y_true[:, 1:]

    # Extract the prediction from student model
    y_pred, y_pred_stu = y_pred[:, :1], y_pred[:, 1:]

    loss = beta * focal_loss(y_true, y_pred) + (1 - beta) * binary_crossentropy(
        y_pred_teacher, y_pred_stu)

    return loss


model = create_model((num_features, 1), 2)
model.compile(loss=knowledge_distillation_fl, optimizer='adam',
              metrics=[auc_2, auc_3])

checkpoint = ModelCheckpoint('student_model_FL.h5', monitor='val_auc_2',
                             verbose=1,
                             save_best_only=True, mode='max',
                             save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_auc_2', factor=0.5, patience=4,
                                   verbose=1, mode='max', epsilon=0.0001)

early = EarlyStopping(monitor="val_auc_2",
                      mode="max",
                      patience=9)

history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=BATCH_SIZE,
                    callbacks=[checkpoint, reduceLROnPlat, early],
                    validation_data=(x_valid, y_valid))

# -------------------------------------------------------------------
## KNOWLEDGE DISTILLATION 3 (KNOWLEDGE CORRECTION WITH LOGIT)

from scipy.special import logit


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


TEMPERATURE = 2

y_knowledge_logit = logit(y_knowledge)
y_temperature = sigmoid(y_knowledge_logit / TEMPERATURE)

x_train, x_valid, y_train, y_valid, y_knowledge_train, y_knowledge_valid = train_test_split(
    train, y, y_temperature,
    stratify=y, test_size=0.2, random_state=SEED)

y_train = np.vstack((y_train[:, 0], y_knowledge_train)).T
y_valid = np.vstack((y_valid[:, 0], y_knowledge_valid)).T

print(y_train.shape)
print(y_train[0])

model = create_model((num_features, 1), 2)
model.compile(
    loss=knowledge_distillation_fl,
    optimizer='adam',
    metrics=[auc_2, auc_3])

checkpoint = ModelCheckpoint(
    './model/student_model_FL.h5', monitor='val_auc_2',
    verbose=1,
    save_best_only=True, mode='max',
    save_weights_only=True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_auc_2', factor=0.5, patience=4,
                                   verbose=1, mode='max', epsilon=0.0001)
early = EarlyStopping(monitor="val_auc_2",
                      mode="max",
                      patience=9)
callbacks_list = [checkpoint, reduceLROnPlat, early]

history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=1024,
                    callbacks=callbacks_list,
                    validation_data=(x_valid, y_valid))

model.load_weights('./model/student_model_FL.h5')
prediction = model.predict(x_valid, batch_size=512, verbose=1)
roc_auc_score(y_valid[:, 0], prediction)

# -------------------------------------------------------------------
## OOF TRAIN AND PREDICT

num_fold = 5
folds = list(
    StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=7).split(
        train, y))

y_test_pred_log = np.zeros(len(train))
y_train_pred_log = np.zeros(len(train))
print(y_test_pred_log.shape)
print(y_train_pred_log.shape)
score = []

for j, (train_idx, valid_idx) in enumerate(folds):
    print('\nFOLD : ', j)
    x_train, x_valid = train[train_idx], train[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    y_knowledge_train, y_knowledge_valid = y_temperature[train_idx], \
                                           y_temperature[valid_idx]

    y_train = np.vstack((y_train, y_knowledge_train)).T
    y_valid = np.vstack((y_valid, y_knowledge_valid)).T

    model = create_model((NUM_FEATURES, 1), 2)

    model.compile(loss=knowledge_distillation_fl,
                  optimizer='adam',
                  metrics=[auc_2, auc_3])

    checkpoint = ModelCheckpoint(
        './model/student_model_fl_fold{}.h5'.format(j),
        monitor='val_auc_2',
        verbose=0,
        save_best_only=True,
        mode='max',
        save_weights_only=True
    )

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_auc_2',
                                       factor=0.5,
                                       patience=4,
                                       verbose=1,
                                       mode='max',
                                       epsilon=0.0001)

    early = EarlyStopping(monitor="val_auc_2",
                          mode="max",
                          patience=9)

    history = model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, reduceLROnPlat, early],
        validation_data=(x_valid, y_valid)
    )

    model.load_weights(
        './model/student_model_fl_fold{}.h5'.format(j))

    prediction = model.predict(x_valid,
                               batch_size=512,
                               verbose=0)
    score.append(roc_auc_score(y_valid[:, 0], prediction[:, 1]))
    prediction = model.predict(test,
                               batch_size=512,
                               verbose=0)

    y_test_pred_log += np.squeeze(prediction[:, 1])

    prediction = model.predict(train,
                               batch_size=512,
                               verbose=0)

    y_train_pred_log += np.squeeze(prediction[:, 1])

    del x_train, x_valid, y_train, y_valid, y_knowledge_train, y_knowledge_valid
    gc.collect()

print("OOF score: ", roc_auc_score(y, y_train_pred_log / num_fold))
print("average {} folds score: ".format(num_fold), np.sum(score) / num_fold)
