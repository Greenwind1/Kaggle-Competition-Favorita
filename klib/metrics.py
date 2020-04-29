# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import cohen_kappa_score

import tensorflow as tf
import keras.backend as K
from keras.callbacks import Callback


def cohen_kappa(y_pred, y_true,
                y_pow=2,
                eps=1e-16,
                N=5,
                batch_size=32,
                name='cohen_kappa'):
    """
    A continuous differentiable approximation of discrete kappa loss.
        Args:
            y_pred: 2D tensor or array, [batch_size, num_classes]
            y_true: 2D tensor or array,[batch_size, num_classes]
            y_pow: int,  e.g. y_pow=2
            N: typically num_classes of the model
            batch_size: batch_size of the training or validation ops
            eps: a float, prevents divide by zero
            name: Optional scope/name for op_scope.
        Returns:
            A tensor with the kappa loss.
    """

    with tf.name_scope(name):
        y_true = tf.to_float(y_true)
        repeat_op = tf.to_float(
            tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N])
        )
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.to_float((N - 1) ** 2)

        pred_ = y_pred ** y_pow
        try:
            pred_norm = pred_ / (
                    eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (
                    eps + tf.reshape(tf.reduce_sum(pred_, 1), [batch_size, 1]))

        hist_rater_a = tf.reduce_sum(pred_norm, 0)
        hist_rater_b = tf.reduce_sum(y_true, 0)

        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)

        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(weights * tf.matmul(
            tf.reshape(hist_rater_a, [N, 1]),
            tf.reshape(hist_rater_b, [1, N])
        ) / tf.to_float(batch_size))

        return nom / (denom + eps)


def cohen_kappa_wrapped(y_pow=2, eps=1e-16, N=5, batch_size=128):
    def lossy_qwk(y_true, y_pred):
        return cohen_kappa(y_true, y_pred,
                           y_pow=y_pow, eps=eps, N=N, batch_size=batch_size)

    return lossy_qwk


class QWKEvaluation(Callback):
    def __init__(self,
                 validation_data=(), batch_size=64, interval=1, savedir='.'):
        super(Callback, self).__init__()

        self.interval = interval
        self.batch_size = batch_size
        self.valid_generator, self.y_val = validation_data
        self.history = []
        self.savedir = '{}_best-qwk.h5'.format(savedir)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_generator(
                generator=self.valid_generator,
                steps=np.ceil(float(len(self.y_val)) / float(self.batch_size)),
                workers=1,
                use_multiprocessing=False,
                verbose=1)

            def flatten(y):
                return np.argmax(y, axis=1).reshape(-1)

            score = cohen_kappa_score(flatten(self.y_val),
                                      flatten(y_pred),
                                      labels=[0, 1, 2, 3, 4],
                                      weights='quadratic')

            print("\n epoch: %d - QWK_score: %.6f \n" % (epoch + 1, score))

            self.history.append(score)

            if score >= max(self.history):
                print('saving checkpoint: {:.5f}'.format(score))
                self.model.save_weights(self.savedir)


class QWKEvaluation_Ordinal(Callback):
    def __init__(self,
                 validation_data=(), batch_size=64, interval=1, savedir='.'):
        super(Callback, self).__init__()

        self.interval = interval
        self.batch_size = batch_size
        self.valid_generator, self.y_val = validation_data
        self.history = []
        self.savedir = '{}_best-qwk.h5'.format(savedir)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_generator(
                generator=self.valid_generator,
                steps=np.ceil(float(len(self.y_val)) / float(self.batch_size)),
                workers=1,
                use_multiprocessing=False,
                verbose=1)

            def flatten(y):
                return (y >= 0.5).sum(axis=1).reshape(-1)

            score = cohen_kappa_score(flatten(self.y_val),
                                      flatten(y_pred),
                                      labels=[0, 1, 2, 3, 4],
                                      weights='quadratic')

            print("\n epoch: %d - QWK_score: %.6f \n" % (epoch + 1, score))

            self.history.append(score)

            if score >= max(self.history):
                print('saving checkpoint: {:.5f}'.format(score))
                self.model.save_weights(self.savedir)


class QWKEvaluation_regression(Callback):
    def __init__(self,
                 validation_data=(), batch_size=64, interval=1, savedir='.'):
        super(Callback, self).__init__()

        self.interval = interval
        self.batch_size = batch_size
        self.valid_generator, self.y_val = validation_data
        self.history = []
        self.savedir = '{}_best-qwk.h5'.format(savedir)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_generator(
                generator=self.valid_generator,
                steps=np.ceil(float(len(self.y_val)) / float(self.batch_size)),
                workers=1,
                use_multiprocessing=False,
                verbose=1)

            pred_rounded = np.clip(y_pred.round(0), 0, 4).astype(int)
            score = cohen_kappa_score(self.y_val,
                                      pred_rounded,
                                      labels=[0, 1, 2, 3, 4],
                                      weights='quadratic')

            print("\n epoch: %d - QWK_score: %.6f \n" % (epoch + 1, score))

            self.history.append(score)

            if score >= max(self.history):
                print('saving checkpoint: {:.5f}'.format(score))

                ## if using pop(0) in defining model, it will return an error.
                ## https://github.com/keras-team/keras/issues/2790
                # self.model.save(self.savedir)

                self.model.save_weights(self.savedir)


def modified_mse(y_true, y_pred):
    """
    Custom Loss Function to clip targets in [0, 4]
    all variables in this function except figures must be tensors.
    https://blog.shikoan.com/keras-backend-greater-less/
    ---
    :param y_true: tensor
    :param y_pred: tensor
    :return: modified mse tensor
    y_true = K.variable(np.array([[1], [0], [3]]), np.float32)
    y_pred = K.variable(np.array([[0.5], [-0.5], [3.5]]), np.float32)
    normal mse = 0.25
    modified mse ~ 0.17
    """
    y_pred = K.clip(y_pred, 0, 4)
    # print(K.eval(y_pred))
    mse_tensor = K.mean(K.square(y_pred - y_true))
    # print(K.eval(mse_tensor))
    return mse_tensor


def modified_mse2(y_true, y_pred):
    """
    Custom Loss Function to clip targets in [0, 4]
    all variables in this function except figures must be tensors.
    https://blog.shikoan.com/keras-backend-greater-less/
    ---
    :param y_true: tensor
    :param y_pred: tensor
    :return: modified mse tensor
    y_true = K.variable(np.array([[1], [0], [3]]), np.float32)
    y_pred = K.variable(np.array([[0.5], [-0.5], [3.5]]), np.float32)
    normal mse = 0.25
    modified mse ~ 0.17
    """
    mse_tensor1 = K.mean(K.square(y_pred - y_true))
    y_pred = K.clip(y_pred, 0, 4)
    # print(K.eval(y_pred))
    mse_tensor2 = K.mean(K.square(y_pred - y_true))
    # print(K.eval(mse_tensor))
    mse_tensor = (mse_tensor1 + mse_tensor2) / 2
    return mse_tensor


def ks_log_loss(y_true, y_pred):
    """
    Keras objective log_loss
    :param y_true: array, actual
    :param y_pred: array, predicted
    :return: scalar value
    """
    return np.mean(
        -(y_true * np.log(y_pred + 1e-16) + (1 - y_true) * np.log(
            1 - y_pred + 1e-16))
    )
