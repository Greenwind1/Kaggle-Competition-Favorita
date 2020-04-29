# -*- coding:utf-8 -*-
# 04.Apr.2018 default auto encoder
# 04.Apr.2018 added activation regularizer
# 05.Apr.2018 adding drop out instead of regularizer
# 05.Apr.2018 chaged to CNN auto encoder
# -------------------------------------------------------------------
# * To launch tensorboard, type following command.
# http::/0.0.0.0:6006
# > tensorboard --logdir=/tmp/autoencoder

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

from keras import backend as K
from keras.layers import Input, Dense
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, TensorBoard
from matplotlib.pyplot import cm
from keras import regularizers

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# functional API coding ---------------------------------------------

input_img = Input(shape=(28, 28, 1,))
x = Conv2D(filters=16,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation='relu')(input_img)
x = MaxPooling2D(pool_size=(2, 2),
                 padding='same')(x)
x = Conv2D(filters=8,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2),
                 padding='same')(x)
x = Conv2D(filters=8,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation='relu')(x)
encoded = MaxPooling2D(pool_size=(2, 2),
                       padding='same')(x)
# at this stage, image is converted to (4, 4, 8) tensor.
x = Conv2D(filters=8,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation='relu')(encoded)
x = UpSampling2D(size=(2, 2))(x)
x = Conv2D(filters=8,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation='relu')(x)
x = UpSampling2D(size=(2, 2))(x)
x = Conv2D(filters=16,
           kernel_size=(3, 3),
           strides=(1, 1),
           activation='relu')(x)
x = UpSampling2D(size=(2, 2))(x)
decoded = Conv2D(filters=1,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='sigmoid')(x)

encoder = Model(inputs=input_img, outputs=encoded)
auto_encoder = Model(inputs=input_img, outputs=decoded)
auto_encoder.compile(optimizer='adadelta',
                     loss='binary_crossentropy')

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255
X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
X_test = X_test.astype('float32') / 255
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))

auto_encoder.fit(x=X_train, y=X_train,
                 epochs=50,
                 callbacks=[EarlyStopping(patience=5),
                            TensorBoard('./logs')],
                 batch_size=256,
                 validation_data=(X_test, X_test),
                 shuffle=True)

score = auto_encoder.evaluate(x=X_test, y=X_test)
print('binary cross entropy : {}'.format(round(score, 3)))

encoded_imgs = encoder.predict(X_test)
decoded_imgs = auto_encoder.predict(X_test)

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
fig.suptitle('Raw vs Decoded with CNN')
fig.savefig('./mnist_raw_vs_decoded_with_CNN.png',
            dpi=140)

fig, axes = plt.subplots(1, n, figsize=(2 * n, 8))
for i in range(0, n):
    axes[i].imshow(encoded_imgs[i].reshape(4, 4 * 8).T,
                   cmap=cm.pink)
    axes[i].grid(False)
fig.suptitle('Encoded with CNN')
fig.savefig('./mnist_encoded_with_CNN.png',
            dpi=140)
