# -*- coding: utf-8 -*-
import numpy as np
import cv2

import tensorflow as tf
import keras as kr
import keras.backend as K
from keras.utils import Sequence
from keras.layers import (Input, Dense, BatchNormalization, Flatten,
                          Conv1D, Conv2D, GlobalAveragePooling1D,
                          GlobalAveragePooling2D, concatenate)
from keras.models import Model
from keras.optimizers import Adam
from keras.objectives import binary_crossentropy
from keras.metrics import categorical_crossentropy

from config import Config

"""
Sequence
- https://keras.io/utils/#sequence
- Sequence class must implement `__getitem__` and `__len__` methods. 
  If you want to modify your dataset between epochs, 
  you may implement `on_epoch_end`. 
  The method `__getitem__` should return a complete batch.  
"""

config = Config()


# -------------------------------------------------------------------
#   Image PreProcessing Functions
# -------------------------------------------------------------------
# Image Normalizer

def img_normalize(img, mode=3):
    """ Normalize image array
    :param img: image array, batch array
    :param mode: int, 1: 0 ~ 1, 2: -1 ~ +1, 3: imagenet normalization,
    Others: return original image array
    """
    if mode == 1:
        img = img / 255
        return img
    elif mode == 2:
        img = img / 127.5
        img = img - 1
        return img
    elif mode == 3:
        img = img / 255
        # Here it's ImageNet statistics
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Considering an ordering NCHW (batch, height, width, channel)
        for i in range(3):
            img[..., i] -= mean[i]
            img[..., i] /= std[i]
        return img
    else:
        return img


# -------------------------------------------------------------------
#   GeneratorI
#   - - - - - - - - - - - - -
#   Train : aug = ...
#   Validation : aug = 'None'
#   Predict(TTA) : aug = ...
# -------------------------------------------------------------------

# noinspection PyTypeChecker
class GeneratorI(Sequence):
    """ < Methods Hierarchy >
    1. __len__
    2. __getitem__
            |
            --- __data_generation
                    |
                    --- load_batch_imgs
                        |
                        --- load_one_img
                            |
                            --- preprocess
                            |
                            --- aug (albumentation)
    3. on_epoch_end: initialization when one epoch ends.
    """

    def __init__(self,
                 img_fnames,
                 img_array=None,
                 y_ohe=None,
                 batch_size=32,
                 aug=None,
                 mode='train',
                 img_preprocess1=None,
                 img_preprocess2='None',
                 is_npy=True):
        """
        :param img_fnames: file names array or list like
        :param img_array: image array (N, row, col, channel)
        :param y_ohe: target labels, values for training
        :param batch_size: int
        :param aug: albumentation aug class
        :param img_preprocess1: 1, 2, 3 or None
        :param img_preprocess2: 'None', 'clahe', 'eq_hist'
        :param mode: 'train', 'valid', 'predict'
        :param is_npy: boolean
        """
        self.img_fnames = img_fnames
        self.img_arr = img_array
        self.y_ohe = y_ohe
        self.batch_size = batch_size
        self.aug = aug
        self.mode = mode
        self.img_p1 = img_preprocess1
        self.img_p2 = img_preprocess2
        self.is_npy = is_npy
        self.dir = config.tr_dir
        self.on_epoch_end()

    def preprocess(self, im):
        im = img_normalize(im, mode=self.img_p1)
        return im

    def __len__(self):
        return int(np.ceil(len(self.img_fnames) / self.batch_size))

    def __getitem__(self, index):
        batch_idxs = \
            self.indices[
            (index * self.batch_size):((index + 1) * self.batch_size)
            ]
        return self._data_generation(batch_idxs)

    def load_one_img(self, img_fname):
        # image (jpg, png, ...)
        if not self.is_npy:
            # load image
            img_fname = self.dir + img_fname
            x = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)

            # preprocessing 1
            if self.img_p1:
                x = self.preprocess(x)
            else:
                pass

            # preprocessing 2
            if self.img_p2 == 'None':
                ...
            elif self.img_p2 == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
                x = np.array([clahe.apply(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))
            elif self.img_p2 == 'eq_hist':
                x = np.array([cv2.equalizeHist(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))

            # augmentation with albumentation
            # https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb
            if self.aug is not None:
                x = self.aug(image=x)['image']

            return x

        # numpy.ndarray
        else:
            # -  -  -  -  -  -  -  -  -  -
            # Disk I/O
            # img_fname: str, XXX.npy
            # -  -  -  -  -  -  -  -  -  -
            # img_fname = self.dir + img_fname
            # x = np.load(img_fname)

            # -  -  -  -  -  -  -  -  -  -
            # RAM
            # img_fname: index
            # -  -  -  -  -  -  -  -  -  -
            x = self.img_arr[img_fname]

            # preprocessing 1
            if self.img_p1:
                x = self.preprocess(x)
            else:
                pass

            # preprocessing 2
            if self.img_p2 == 'None':
                ...
            elif self.img_p2 == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
                x = np.array([clahe.apply(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))
            elif self.img_p2 == 'eq_hist':
                x = np.array([cv2.equalizeHist(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))

            # augmentation with albumentation
            # https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb
            if self.aug is not None:
                x = self.aug(image=x)['image']

            return x

    def load_batch_imgs(self, img_fnames):
        batch_data = [self.load_one_img(x) for x in img_fnames]
        return np.array(batch_data)

    def _data_generation(self, batch_idxs):
        if self.mode != 'predict':
            # batch_x
            batch_x = self.load_batch_imgs([
                self.img_fnames[i] for i in batch_idxs
            ])
            # batch_y
            batch_y = self.y_ohe[batch_idxs, ...]
            return (batch_x, batch_y)

        else:
            # batch_x
            batch_x = self.load_batch_imgs([
                self.img_fnames[i] for i in batch_idxs
            ])
            return batch_x

    def on_epoch_end(self):
        # initialize the indices
        self.indices = np.arange(len(self.img_fnames))

        if self.mode == 'train':
            # shuffle the indices
            np.random.shuffle(self.indices)


# -------------------------------------------------------------------
#   GeneratorII
#   - - - - - - - - - - - - -
#   Train : aug = ...
#   Validation : aug = 'None'
#   Predict(TTA) : aug = ...
# -------------------------------------------------------------------

# noinspection PyTypeChecker
class GeneratorII(Sequence):
    """ < Methods Hierarchy >
    1. __len__
    2. __getitem__
            |
            --- __data_generation
                    |
                    --- load_batch_imgs
                        |
                        --- load_one_img
                            |
                            --- preprocess
                            |
                            --- aug (albumentation)
    3. on_epoch_end: initialization when one epoch ends.
    """

    def __init__(self,
                 img_fnames,
                 img_array=None,
                 y_ohe=None,
                 batch_size=32,
                 aug=None,
                 mode='train',
                 img_preprocess1=None,
                 img_preprocess2='None',
                 is_npy=True):
        """
        :param img_fnames: file names array or list like
        :param img_array: image array (N, row, col, channel)
        :param y_ohe: target labels, values for training
        :param batch_size: int
        :param aug: albumentation aug class
        :param img_preprocess1: function
        :param img_preprocess2: 'None', 'clahe', 'eq_hist'
        :param mode: 'train', 'valid', 'predict'
        :param is_npy: boolean
        """
        self.img_fnames = img_fnames
        self.img_arr = img_array
        self.y_ohe = y_ohe
        self.batch_size = batch_size
        self.aug = aug
        self.mode = mode
        self.img_p1 = img_preprocess1
        self.img_p2 = img_preprocess2
        self.is_npy = is_npy
        self.dir = config.tr_dir
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.img_fnames) / self.batch_size))

    def __getitem__(self, index):
        batch_idxs = \
            self.indices[
            (index * self.batch_size):((index + 1) * self.batch_size)
            ]
        return self._data_generation(batch_idxs)

    def load_one_img(self, img_fname):
        # image (jpg, png, ...)
        if not self.is_npy:
            # load image
            img_fname = self.dir + img_fname
            x = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)

            # preprocessing 1
            if self.img_p1:
                x = self.img_p1(x)
            else:
                pass

            # preprocessing 2
            if self.img_p2 == 'None':
                ...
            elif self.img_p2 == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
                x = np.array([clahe.apply(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))
            elif self.img_p2 == 'eq_hist':
                x = np.array([cv2.equalizeHist(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))

            # augmentation with albumentation
            # https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb
            if self.aug is not None:
                x = self.aug(image=x)['image']

            return x

        # numpy.ndarray
        else:
            # -  -  -  -  -  -  -  -  -  -
            # Disk I/O
            # img_fname: str, XXX.npy
            # -  -  -  -  -  -  -  -  -  -
            # img_fname = self.dir + img_fname
            # x = np.load(img_fname)

            # -  -  -  -  -  -  -  -  -  -
            # RAM
            # img_fname: index
            # -  -  -  -  -  -  -  -  -  -
            x = self.img_arr[img_fname]

            # preprocessing 1
            if self.img_p1:
                x = self.preprocess(x)
            else:
                pass

            # preprocessing 2
            if self.img_p2 == 'None':
                ...
            elif self.img_p2 == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
                x = np.array([clahe.apply(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))
            elif self.img_p2 == 'eq_hist':
                x = np.array([cv2.equalizeHist(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))

            # augmentation with albumentation
            # https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb
            if self.aug is not None:
                x = self.aug(image=x)['image']

            return x

    def load_batch_imgs(self, img_fnames):
        batch_data = [self.load_one_img(x) for x in img_fnames]
        return np.array(batch_data)

    def _data_generation(self, batch_idxs):
        if self.mode != 'predict':
            # batch_x
            batch_x = self.load_batch_imgs([
                self.img_fnames[i] for i in batch_idxs
            ])
            # batch_y
            batch_y = self.y_ohe[batch_idxs, ...]
            return (batch_x, batch_y)

        else:
            # batch_x
            batch_x = self.load_batch_imgs([
                self.img_fnames[i] for i in batch_idxs
            ])
            return batch_x

    def on_epoch_end(self):
        # initialize the indices
        self.indices = np.arange(len(self.img_fnames))

        if self.mode == 'train':
            # shuffle the indices
            np.random.shuffle(self.indices)


# -------------------------------------------------------------------
#   TrainGeneratorPreprocessVIII
#   PredGeneratorPreprocessVIII
#   - - - - - - - - - - - - -
#   Train : aug = ...
#   Validation : aug = None
#   TTA : aug = ...
#   - - - - - - - - - - - - -
#   .npy loading
#   CutMix
#   DRD data, fast crop, w/o padding
# -------------------------------------------------------------------

class TrainGeneratorPreprocessVIII(Sequence):
    def __init__(self,
                 img_fnames,
                 y_ohe=None,
                 batch_size=32,
                 aug=None,
                 cutmix_p=0.5,
                 cutmix_alpha=0.5,
                 img_preprocess='None',
                 isvalid=False,
                 is_preprocess=True,
                 is_npy=True):
        """
        :param img_fnames: file names array or list like
        :param batch_size: int
        :param aug: albumentation aug class
        :param cutmix_p: [0, 1] lower value means using raw images w/o CutMix
        :param img_preprocess: 'None', 'clahe', 'eq_hist'
        :param isvalid: boolean
        """
        self.img_fnames = img_fnames
        self.y_ohe = y_ohe
        self.batch_size = batch_size
        self.aug = aug
        self.cutmix_p = cutmix_p
        self.cutmix_alpha = cutmix_alpha
        self.isvalid = isvalid
        self.img_p = img_preprocess
        self.ispreprocess = is_preprocess
        self.is_npy = is_npy
        self.on_epoch_end()

    @staticmethod
    def preprocess(im, th=7, size=512):
        gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        mask = gray_im > th
        check_shape = im[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            ...
        else:
            im = np.array(
                [im[..., i][np.ix_(mask.any(1), mask.any(0))] for i in
                 range(im.shape[2])]
            ).transpose((1, 2, 0))

        im = cv2.resize(im, dsize=(size, size))

        return im

    def __len__(self):
        return int(np.ceil(len(self.img_fnames) / self.batch_size))

    def __getitem__(self, index):
        batch_idxs = \
            self.indices[
            (index * self.batch_size):((index + 1) * self.batch_size)
            ]
        batch_idxs_cm = \
            self.cutmix_indices[
            (index * self.batch_size):((index + 1) * self.batch_size)
            ]
        batch_cm_lambdas = \
            self.cutmix_lambdas[
            (index * self.batch_size):((index + 1) * self.batch_size)
            ]
        return self._data_generation(batch_idxs,
                                     batch_idxs_cm,
                                     batch_cm_lambdas)

    def load_one_img(self, img_fname):
        if not self.is_npy:
            if '_right' in img_fname or '_left' in img_fname:
                img_fname = config.tr_drd_dir + img_fname
            elif 'IDRiD' in img_fname:
                img_fname = config.tr_drd_idrid + \
                            img_fname.split('_')[0] + \
                            '/' + img_fname.split('_')[1] + \
                            '_' + img_fname.split('_')[2]
            else:
                img_fname = config.tr_dir + img_fname
            x = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)

            if self.ispreprocess:
                x = self.preprocess(x, size=config.img_size)
            else:
                x = cv2.resize(x, (config.img_size, config.img_size))

            if self.img_p == 'None':
                ...
            elif self.img_p == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
                x = np.array([clahe.apply(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))
            elif self.img_p == 'eq_hist':
                x = np.array([cv2.equalizeHist(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))

            # augmentation with albumentation
            # https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb
            if self.aug is not None:
                x = self.aug(image=x)['image']
            x = x / (x.max() + 1e-32)
            return x
        else:
            if '_right' in img_fname or '_left' in img_fname:
                img_fname = config.tr_drd_dir + img_fname
            elif 'IDRiD' in img_fname:
                img_fname = config.tr_drd_idrid + \
                            img_fname.split('_')[0] + \
                            '/' + img_fname.split('_')[1] + \
                            '_' + img_fname.split('_')[2]
            else:
                img_fname = config.tr_dir + img_fname

            x = np.load(img_fname)
            # x = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
            #
            # if self.ispreprocess:
            #     x = self.preprocess(x, size=config.img_size)
            # else:
            #     x = cv2.resize(x, (config.img_size, config.img_size))

            if self.img_p == 'None':
                ...
            elif self.img_p == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
                x = np.array([clahe.apply(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))
            elif self.img_p == 'eq_hist':
                x = np.array([cv2.equalizeHist(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))

            # augmentation with albumentation
            # https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb
            if self.aug is not None:
                x = self.aug(image=x)['image']
            x = x / (x.max() + 1e-32)
            return x

    def load_batch_imgs(self, img_fnames):
        batch_data = [self.load_one_img(x) for x in img_fnames]
        return np.array(batch_data)

    def make_mask(self, cm_lambda):
        mask = np.ones((config.img_size, config.img_size, config.channel))
        rx, ry = np.random.randint(0, config.img_size, 2)
        rwh = int(config.img_size * np.sqrt(1 - cm_lambda))
        rxw = rx + rwh
        ryw = ry + rwh
        mask[rx:rxw, ry:ryw, ...] = 0
        return mask

    def make_masks(self, batch_cm_lambdas):
        masks = [self.make_mask(cm_lambda=l) for l in batch_cm_lambdas]
        return np.array(masks)

    def _data_generation(self, batch_idxs, batch_idxs_cm, batch_cm_lambdas):
        if self.isvalid:
            # batch_x
            batch_x = self.load_batch_imgs([
                self.img_fnames[i] for i in batch_idxs
            ])
            # batch_y
            batch_y = self.y_ohe[batch_idxs, ...]
            return (batch_x, batch_y)
        else:
            # w/o CutMix
            if np.random.rand(1) > self.cutmix_p:
                # batch_x
                batch_x = self.load_batch_imgs([
                    self.img_fnames[i] for i in batch_idxs
                ])
                # batch_y
                batch_y = self.y_ohe[batch_idxs, ...]
                return (batch_x, batch_y)
            # CutMix
            else:
                # shape: (batch, w, h, ch)
                # batch_x
                batch_x1 = self.load_batch_imgs([
                    self.img_fnames[i] for i in batch_idxs
                ])
                batch_x2 = self.load_batch_imgs([
                    self.img_fnames[i] for i in batch_idxs_cm
                ])
                batch_mask = self.make_masks(batch_cm_lambdas)
                batch_x = batch_x1 * batch_mask + \
                          batch_x2 * (batch_mask == 0).astype(int)

                # batch_y
                batch_y1 = self.y_ohe[batch_idxs, ...]
                batch_y2 = self.y_ohe[batch_idxs_cm, ...]
                batch_y = (batch_y1 * batch_cm_lambdas) + \
                          (batch_y2 * (1 - batch_cm_lambdas))

                return (batch_x, batch_y)

    def on_epoch_end(self):
        # initialize the indices and lambdas
        self.indices = np.arange(len(self.img_fnames))
        self.cutmix_indices = np.arange(len(self.img_fnames))

        # generate cutmix lambda, [l0, l1, ..., ln]
        self.cutmix_lambdas = np.random.beta(
            self.cutmix_alpha,
            self.cutmix_alpha,
            len(self.img_fnames)
        )

        # on Training
        if not self.isvalid:
            # shuffle the indices
            np.random.shuffle(self.indices)
            np.random.shuffle(self.cutmix_indices)


class PredGeneratorPreprocessVIII(Sequence):
    def __init__(self,
                 img_fnames,
                 batch_size=32,
                 aug=None,
                 img_preprocess='None',
                 istest=False,
                 is_preprocess=True,
                 is_npy=True):
        """
        :param img_fnames:
        :param batch_size:
        :param aug:
        :param img_preprocess: 'None', 'clahe', 'eq_hist'
        :param istest:
        """
        self.img_fnames = img_fnames
        self.batch_size = batch_size
        self.aug = aug
        self.img_p = img_preprocess
        self.istest = istest
        self.ispreprocess = is_preprocess
        self.is_npy = is_npy
        self.on_epoch_end()

    @staticmethod
    def preprocess(im, th=7, size=512):
        gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        mask = gray_im > th
        check_shape = im[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            ...
        else:
            im = np.array(
                [im[..., i][np.ix_(mask.any(1), mask.any(0))] for i in
                 range(im.shape[2])]
            ).transpose((1, 2, 0))

        im = cv2.resize(im, dsize=(size, size))

        return im

    def __len__(self):
        return int(np.ceil(len(self.img_fnames) / self.batch_size))

    def __getitem__(self, index):
        batch_idxs = \
            self.indices[
            (index * self.batch_size):((index + 1) * self.batch_size)
            ]
        return self._data_generation(batch_idxs)

    def load_one_img(self, img_fname):
        if not self.is_npy:
            if self.istest:
                img_fname = config.te_dir + img_fname
            else:
                if '_right' in img_fname or '_left' in img_fname:
                    img_fname = config.tr_drd_dir + img_fname
                elif 'IDRiD' in img_fname:
                    img_fname = config.tr_drd_idrid + \
                                img_fname.split('_')[0] + \
                                '/' + img_fname.split('_')[1] + \
                                '_' + img_fname.split('_')[2]
                else:
                    img_fname = config.tr_dir + img_fname
            x = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
            if self.ispreprocess:
                x = self.preprocess(x, size=config.img_size)
            else:
                x = cv2.resize(x, (config.img_size, config.img_size))

            if self.img_p == 'None':
                ...
            elif self.img_p == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
                x = np.array([clahe.apply(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))
            elif self.img_p == 'eq_hist':
                x = np.array([cv2.equalizeHist(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))

            # augmentation with albumentation
            # https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb
            if self.aug is not None:
                x = self.aug(image=x)['image']
            x = x / (x.max() + 1e-32)
            return x
        else:
            if '_right' in img_fname or '_left' in img_fname:
                img_fname = config.tr_drd_dir + img_fname
            elif 'IDRiD' in img_fname:
                img_fname = config.tr_drd_idrid + \
                            img_fname.split('_')[0] + \
                            '/' + img_fname.split('_')[1] + \
                            '_' + img_fname.split('_')[2]
            else:
                img_fname = config.tr_dir + img_fname

            x = np.load(img_fname)
            # x = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
            #
            # if self.ispreprocess:
            #     x = self.preprocess(x, size=config.img_size)
            # else:
            #     x = cv2.resize(x, (config.img_size, config.img_size))

            if self.img_p == 'None':
                ...
            elif self.img_p == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
                x = np.array([clahe.apply(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))
            elif self.img_p == 'eq_hist':
                x = np.array([cv2.equalizeHist(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))

            # augmentation with albumentation
            # https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb
            if self.aug is not None:
                x = self.aug(image=x)['image']
            x = x / (x.max() + 1e-32)
            return x

    def load_batch_imgs(self, img_fnames):
        batch_data = [self.load_one_img(x) for x in img_fnames]
        return np.array(batch_data)

    def _data_generation(self, batch_idxs):
        # batch_x
        batch_x = self.load_batch_imgs([
            self.img_fnames[i] for i in batch_idxs
        ])
        return batch_x

    def on_epoch_end(self):
        # initialize the indices
        self.indices = np.arange(len(self.img_fnames))

        # shuffle the indices
        # np.random.shuffle(self.indices)


# -------------------------------------------------------------------
#   TrainGeneratorPreprocessVIIII
#   PredGeneratorPreprocessVIIII
#   - - - - - - - - - - - - -
#   Train : aug = ...
#   Validation : aug = None
#   TTA : aug = ...
#   - - - - - - - - - - - - -
#   .npy loading
#   CutMix with surface ratio
#   DRD data, fast crop, w/o padding
# -------------------------------------------------------------------

class TrainGeneratorPreprocessVIIII(Sequence):
    def __init__(self,
                 img_fnames,
                 y_ohe=None,
                 batch_size=32,
                 aug=None,
                 cutmix_p=0.5,
                 cutmix_alpha=0.5,
                 img_preprocess='None',
                 isvalid=False,
                 is_preprocess=True,
                 is_npy=True):
        """
        :param img_fnames: file names array or list like
        :param batch_size: int
        :param aug: albumentation aug class
        :param cutmix_p: [0, 1] lower value means using raw images w/o CutMix
        :param img_preprocess: 'None', 'clahe', 'eq_hist'
        :param isvalid: boolean
        """
        self.img_fnames = img_fnames
        self.y_ohe = y_ohe
        self.batch_size = batch_size
        self.aug = aug
        self.cutmix_p = cutmix_p
        self.cutmix_alpha = cutmix_alpha
        self.isvalid = isvalid
        self.img_p = img_preprocess
        self.ispreprocess = is_preprocess
        self.is_npy = is_npy
        self.on_epoch_end()

    @staticmethod
    def preprocess(im, th=7, size=512):
        gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        mask = gray_im > th
        check_shape = im[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            ...
        else:
            im = np.array(
                [im[..., i][np.ix_(mask.any(1), mask.any(0))] for i in
                 range(im.shape[2])]
            ).transpose((1, 2, 0))

        im = cv2.resize(im, dsize=(size, size))

        return im

    def __len__(self):
        return int(np.ceil(len(self.img_fnames) / self.batch_size))

    def __getitem__(self, index):
        batch_idxs = \
            self.indices[
            (index * self.batch_size):((index + 1) * self.batch_size)
            ]
        batch_idxs_cm = \
            self.cutmix_indices[
            (index * self.batch_size):((index + 1) * self.batch_size)
            ]
        batch_cm_lambdas = \
            self.cutmix_lambdas[
            (index * self.batch_size):((index + 1) * self.batch_size)
            ]
        return self._data_generation(batch_idxs,
                                     batch_idxs_cm,
                                     batch_cm_lambdas)

    def load_one_img(self, img_fname):
        if not self.is_npy:
            if '_right' in img_fname or '_left' in img_fname:
                img_fname = config.tr_drd_dir + img_fname
            elif 'IDRiD' in img_fname:
                img_fname = config.tr_drd_idrid + \
                            img_fname.split('_')[0] + \
                            '/' + img_fname.split('_')[1] + \
                            '_' + img_fname.split('_')[2]
            else:
                img_fname = config.tr_dir + img_fname
            x = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)

            if self.ispreprocess:
                x = self.preprocess(x, size=config.img_size)
            else:
                x = cv2.resize(x, (config.img_size, config.img_size))

            if self.img_p == 'None':
                ...
            elif self.img_p == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
                x = np.array([clahe.apply(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))
            elif self.img_p == 'eq_hist':
                x = np.array([cv2.equalizeHist(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))

            # augmentation with albumentation
            # https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb
            if self.aug is not None:
                x = self.aug(image=x)['image']
            x = x / (x.max() + 1e-32)
            return x
        else:
            if '_right' in img_fname or '_left' in img_fname:
                img_fname = config.tr_drd_dir + img_fname
            elif 'IDRiD' in img_fname:
                img_fname = config.tr_drd_idrid + \
                            img_fname.split('_')[0] + \
                            '/' + img_fname.split('_')[1] + \
                            '_' + img_fname.split('_')[2]
            else:
                img_fname = config.tr_dir + img_fname

            x = np.load(img_fname)
            # x = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
            #
            # if self.ispreprocess:
            #     x = self.preprocess(x, size=config.img_size)
            # else:
            #     x = cv2.resize(x, (config.img_size, config.img_size))

            if self.img_p == 'None':
                ...
            elif self.img_p == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
                x = np.array([clahe.apply(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))
            elif self.img_p == 'eq_hist':
                x = np.array([cv2.equalizeHist(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))

            # augmentation with albumentation
            # https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb
            if self.aug is not None:
                x = self.aug(image=x)['image']
            x = x / (x.max() + 1e-32)
            return x

    def load_batch_imgs(self, img_fnames):
        batch_data = [self.load_one_img(x) for x in img_fnames]
        return np.array(batch_data)

    def make_mask(self, cm_lambda):
        mask = np.ones((config.img_size, config.img_size, config.channel))
        rx, ry = np.random.randint(0, config.img_size, 2)
        rwh = int(config.img_size * np.sqrt(1 - cm_lambda))
        rxw = rx + rwh
        ryw = ry + rwh
        mask[rx:rxw, ry:ryw, ...] = 0
        return mask

    def make_masks(self, batch_cm_lambdas):
        masks = [self.make_mask(cm_lambda=l) for l in batch_cm_lambdas]
        return np.array(masks)

    def _data_generation(self, batch_idxs, batch_idxs_cm, batch_cm_lambdas):
        if self.isvalid:
            # batch_x
            batch_x = self.load_batch_imgs([
                self.img_fnames[i] for i in batch_idxs
            ])
            # batch_y
            batch_y = self.y_ohe[batch_idxs, ...]
            return (batch_x, batch_y)
        else:
            # w/o CutMix
            if np.random.rand(1) > self.cutmix_p:
                # batch_x
                batch_x = self.load_batch_imgs([
                    self.img_fnames[i] for i in batch_idxs
                ])
                # batch_y
                batch_y = self.y_ohe[batch_idxs, ...]
                return (batch_x, batch_y)
            # CutMix
            else:
                # shape: (batch, w, h, ch)
                # batch_x
                batch_x1 = self.load_batch_imgs([
                    self.img_fnames[i] for i in batch_idxs
                ])
                batch_x2 = self.load_batch_imgs([
                    self.img_fnames[i] for i in batch_idxs_cm
                ])
                batch_mask = self.make_masks(batch_cm_lambdas)
                batch_x = batch_x1 * batch_mask + \
                          batch_x2 * (batch_mask == 0).astype(int)

                # batch_y
                batch_y1 = self.y_ohe[batch_idxs, ...]
                batch_y2 = self.y_ohe[batch_idxs_cm, ...]
                batch_cm_sr = batch_mask[..., 0].sum(axis=(1, 2)) / \
                              (batch_mask.shape[1] * batch_mask.shape[2])
                batch_y = (batch_y1 * batch_cm_sr) + \
                          (batch_y2 * (1 - batch_cm_sr))

                return (batch_x, batch_y)

    def on_epoch_end(self):
        # initialize the indices and lambdas
        self.indices = np.arange(len(self.img_fnames))
        self.cutmix_indices = np.arange(len(self.img_fnames))

        # generate cutmix lambda, [l0, l1, ..., ln]
        self.cutmix_lambdas = np.random.beta(
            self.cutmix_alpha,
            self.cutmix_alpha,
            len(self.img_fnames)
        )

        # on Training
        if not self.isvalid:
            # shuffle the indices
            np.random.shuffle(self.indices)
            np.random.shuffle(self.cutmix_indices)


class PredGeneratorPreprocessVIIII(Sequence):
    def __init__(self,
                 img_fnames,
                 batch_size=32,
                 aug=None,
                 img_preprocess='None',
                 istest=False,
                 is_preprocess=True,
                 is_npy=True):
        """
        :param img_fnames:
        :param batch_size:
        :param aug:
        :param img_preprocess: 'None', 'clahe', 'eq_hist'
        :param istest:
        """
        self.img_fnames = img_fnames
        self.batch_size = batch_size
        self.aug = aug
        self.img_p = img_preprocess
        self.istest = istest
        self.ispreprocess = is_preprocess
        self.is_npy = is_npy
        self.on_epoch_end()

    @staticmethod
    def preprocess(im, th=7, size=512):
        gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        mask = gray_im > th
        check_shape = im[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            ...
        else:
            im = np.array(
                [im[..., i][np.ix_(mask.any(1), mask.any(0))] for i in
                 range(im.shape[2])]
            ).transpose((1, 2, 0))

        im = cv2.resize(im, dsize=(size, size))

        return im

    def __len__(self):
        return int(np.ceil(len(self.img_fnames) / self.batch_size))

    def __getitem__(self, index):
        batch_idxs = \
            self.indices[
            (index * self.batch_size):((index + 1) * self.batch_size)
            ]
        return self._data_generation(batch_idxs)

    def load_one_img(self, img_fname):
        if not self.is_npy:
            if self.istest:
                img_fname = config.te_dir + img_fname
            else:
                if '_right' in img_fname or '_left' in img_fname:
                    img_fname = config.tr_drd_dir + img_fname
                elif 'IDRiD' in img_fname:
                    img_fname = config.tr_drd_idrid + \
                                img_fname.split('_')[0] + \
                                '/' + img_fname.split('_')[1] + \
                                '_' + img_fname.split('_')[2]
                else:
                    img_fname = config.tr_dir + img_fname
            x = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
            if self.ispreprocess:
                x = self.preprocess(x, size=config.img_size)
            else:
                x = cv2.resize(x, (config.img_size, config.img_size))

            if self.img_p == 'None':
                ...
            elif self.img_p == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
                x = np.array([clahe.apply(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))
            elif self.img_p == 'eq_hist':
                x = np.array([cv2.equalizeHist(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))

            # augmentation with albumentation
            # https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb
            if self.aug is not None:
                x = self.aug(image=x)['image']
            x = x / (x.max() + 1e-32)
            return x
        else:
            if '_right' in img_fname or '_left' in img_fname:
                img_fname = config.tr_drd_dir + img_fname
            elif 'IDRiD' in img_fname:
                img_fname = config.tr_drd_idrid + \
                            img_fname.split('_')[0] + \
                            '/' + img_fname.split('_')[1] + \
                            '_' + img_fname.split('_')[2]
            else:
                img_fname = config.tr_dir + img_fname

            x = np.load(img_fname)
            # x = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
            #
            # if self.ispreprocess:
            #     x = self.preprocess(x, size=config.img_size)
            # else:
            #     x = cv2.resize(x, (config.img_size, config.img_size))

            if self.img_p == 'None':
                ...
            elif self.img_p == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
                x = np.array([clahe.apply(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))
            elif self.img_p == 'eq_hist':
                x = np.array([cv2.equalizeHist(x[..., i]) for i in
                              range(x.shape[2])]).transpose((1, 2, 0))

            # augmentation with albumentation
            # https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb
            if self.aug is not None:
                x = self.aug(image=x)['image']
            x = x / (x.max() + 1e-32)
            return x

    def load_batch_imgs(self, img_fnames):
        batch_data = [self.load_one_img(x) for x in img_fnames]
        return np.array(batch_data)

    def _data_generation(self, batch_idxs):
        # batch_x
        batch_x = self.load_batch_imgs([
            self.img_fnames[i] for i in batch_idxs
        ])
        return batch_x

    def on_epoch_end(self):
        # initialize the indices
        self.indices = np.arange(len(self.img_fnames))

        # shuffle the indices
        # np.random.shuffle(self.indices)
