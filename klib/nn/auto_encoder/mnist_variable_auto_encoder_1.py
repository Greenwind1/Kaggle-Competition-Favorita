# -*- coding:utf-8 -*-
# 04.Apr.2018 default auto encoder
# 04.Apr.2018 added activation regularizer
# 05.Apr.2018 adding drop out instead of regularizer
# 05.Apr.2018 chaged to CNN auto encoder
# 06.Apr.2018 variable encoder
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
from keras.objectives import binary_crossentropy

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# functional API coding ---------------------------------------------

'''
1. original_dim
2. intermediate_dim
3. latent_dim (mean, log_sigma)
4. latent_dim
5. intermediate_dim
6. original_dim
'''

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

batch_size = 200
original_dim = X_train.shape[1]
intermediate_dim = 256
latent_dim = 2
epochs = 50
epsilon_std = 0.1
patience = 5

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)


def sampling(args):
    global epsilon
    z_mean_s, z_log_sigma_s = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean_s + K.exp(z_log_sigma_s / 2) * epsilon


# latent sampled variable layer
z = Lambda(sampling, output_shape=(latent_dim,))(
    [z_mean, z_log_sigma])

decoder_h = Dense(intermediate_dim, activation='relu')
h_decoded = decoder_h(z)

decoder_mean = Dense(original_dim, activation='sigmoid')
x_decoded_mean = decoder_mean(h_decoded)

# 3 models ----------------------------------------------------------
# vae, encoder and generator
# end-to-end autoencoder
vae = Model(x, x_decoded_mean)

# encoder, from inputs to latent space
encoder = Model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)


# custom loss function (binary_crossentropy and KL divergence)
def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(
        1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma),
        axis=-1)  # axis=-1 : last axis ( average by latent_dim axis )
    return K.mean(xent_loss + kl_loss)  # mean with batch size dim


vae.compile(optimizer='rmsprop', loss=vae_loss)

vae.fit(X_train, X_train,
        shuffle=True,
        epochs=epochs,
        callbacks=[EarlyStopping(patience=patience),
                   TensorBoard('./logs',
                               batch_size=batch_size)],
        batch_size=batch_size,
        validation_data=(X_test, X_test))

x_test_encoded = encoder.predict(X_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.show()
plt.savefig('./vae.png', dpi=140)
