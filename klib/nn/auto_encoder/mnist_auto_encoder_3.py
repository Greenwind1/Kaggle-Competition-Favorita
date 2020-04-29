# -*- coding:utf-8 -*-
# 04.Apr.2018 default auto encoder
# 04.Apr.2018 added activation regularizer
# 05.Apr.2018 adding drop out instead of regularizer

from numpy.random import seed

SEED = 2018
seed(SEED)

from tensorflow import set_random_seed

set_random_seed(SEED)

import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=3,
                        inter_op_parallelism_threads=3,
                        allow_soft_placement=True,
                        device_count={'CPU': 3})
session = tf.Session(config=config)

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
from matplotlib.pyplot import cm
from keras import regularizers

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# functional API coding ---------------------------------------------

encoding_dim = 32
dropout_rate = 0.05
input_img = Input(shape=(784,))
encoded = Dense(units=encoding_dim,
                activation='relu')(input_img)
dropout = Dropout(rate=dropout_rate)(encoded)
decoded = Dense(units=784, activation='sigmoid')(dropout)

auto_encoder = Model(inputs=input_img, outputs=decoded)
encoder = Model(inputs=input_img, outputs=encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_dropout = Dropout(rate=dropout_rate)(encoded_input)
decoder_layer = auto_encoder.layers[-1]
decoder = Model(inputs=encoded_input,
                outputs=decoder_layer(decoder_dropout))

auto_encoder.compile(optimizer='adadelta',
                     loss='binary_crossentropy')

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255
X_train = X_train.reshape(X_train.shape[0],
                          X_train.shape[1] * X_train.shape[2])
X_test = X_test.astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0],
                        X_test.shape[1] * X_test.shape[2])

auto_encoder.fit(x=X_train, y=X_train,
                 epochs=100,
                 callbacks=[EarlyStopping(patience=5)],
                 batch_size=256,
                 validation_data=(X_test, X_test),
                 shuffle=True)

score = auto_encoder.evaluate(x=X_test, y=X_test)
print('binary cross entropy : {}'.format(round(score, 3)))

encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10
fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
for i in range(0, n):
    axes[0, i].imshow(X_test[i].reshape(28, 28),
                      cmap=cm.pink)
    axes[0, i].grid(False)
    axes[1, i].imshow(decoded_imgs[i].reshape(28, 28),
                      cmap=cm.pink)
    axes[1, i].grid(False)
fig.tight_layout()
fig.suptitle('Raw vs Decoded with dropout')
fig.savefig('./mnist_raw_vs_decoded_with_dropout.png',
            dpi=140)
