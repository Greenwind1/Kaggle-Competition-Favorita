import numpy as np
import cv2
import glob
import pandas as pd
import matplotlib.pyplot as plt


def boxfilter(img, r):
    # To be used for the guided filter
    (rows, cols) = img.shape
    imDst = np.zeros_like(img)

    imCum = np.cumsum(img, 0)
    imDst[0: r + 1, :] = imCum[r: 2 * r + 1, :]
    imDst[r + 1: rows - r, :] = \
        imCum[2 * r + 1: rows, :] - imCum[0: rows - 2 * r - 1, :]
    imDst[rows - r: rows, :] = \
        np.tile(imCum[rows - 1, :], [r, 1]) - \
        imCum[rows - 2 * r - 1: rows - r - 1, :]

    imCum = np.cumsum(imDst, 1)
    imDst[:, 0: r + 1] = imCum[:, r: 2 * r + 1]
    imDst[:, r + 1: cols - r] = \
        imCum[:, 2 * r + 1: cols] - imCum[:, 0: cols - 2 * r - 1]
    imDst[:, cols - r: cols] = \
        np.tile(imCum[:, cols - 1], [r, 1]).T - \
        imCum[:, cols - 2 * r - 1: cols - r - 1]

    return imDst


def guidedfilter(I, p, r, eps):
    # Filters p, guided by the image I. Uses r for the box filter radius
    (rows, cols) = I.shape
    N = boxfilter(np.ones([rows, cols]), r)

    meanI = boxfilter(I, r) / N
    meanP = boxfilter(p, r) / N
    meanIp = boxfilter(I * p, r) / N
    covIp = meanIp - meanI * meanP

    meanII = boxfilter(I * I, r) / N
    varI = meanII - meanI * meanI

    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = boxfilter(a, r) / N
    meanB = boxfilter(b, r) / N

    q = meanA * I + meanB
    return q


def faster_dark_channel(img, kernel):
    # Method to evaluate the dark channel faster
    # OpenCV web page: https://bit.ly/2m3CDdO
    tmp = np.amin(img, axis=2)
    return cv2.erode(tmp, kernel)


def atmosphere(img, channel_dark, top_percent):
    R, C, D = img.shape
    # flatten dark to get top thres percentage of bright points.
    # Paper uses thres_percent = 0.1
    flat_dark = channel_dark.ravel()
    req = int((R * C * top_percent) / 100)
    # find indices of top req intensites in dark channed
    indices = np.argpartition(flat_dark, -req)[-req:]

    # flatten image and take max among these pixels
    flat_img = img.reshape(R * C, 3)
    return np.max(flat_img.take(indices, axis=0), axis=0)


def eval_transmission(dark_div, param):
    # returns the estimated transmission
    transmission = 1 - param * dark_div
    return transmission


def depth_map(trans, beta):
    rval = -np.log(trans) / beta
    return rval / np.max(rval)


def radiant_image(image, atmosphere, t, thres):
    R, C, D = image.shape
    temp = np.empty(image.shape)
    t[t < thres] = thres
    for i in range(D):
        temp[:, :, i] = t
    b = (image - atmosphere) / temp + atmosphere
    b[b > 255] = 255
    return b


def haze_rm(orig_img,
            window=30,  # kernel size used by DPC
            top_percent=0.3,
            thres_haze=0.5,
            omega=1.0,
            beta=1.0,
            radius=7,
            eps=0.001):
    # Default
    # window: 50, omega: 0.95, radius: 100, eps: 0.001
    img_gray = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
    img = np.asarray(orig_img, dtype=np.float64)
    img_norm = (img_gray - img_gray.mean()) / (img_gray.max() - img_gray.min())
    kernel = np.ones((window, window), np.float64)
    dark = faster_dark_channel(img, kernel)
    # dark = gen_dark_channel(img, window)
    A = atmosphere(img, dark, top_percent)
    B = img / A

    dark_div = faster_dark_channel(B, kernel)
    t_estimate = eval_transmission(dark_div, omega)
    R, C, _ = img.shape
    t_refined = guidedfilter(img_norm, t_estimate, radius, eps)
    # t_refined = fast_guided_filter(img, t_estimate, radius, eps)
    unhazed = radiant_image(img, A, t_refined, thres_haze)
    depthmap = depth_map(t_refined, beta)
    return [np.array(x, dtype=np.uint8) for x in
            [t_estimate * 255, t_refined * 255, unhazed, dark, depthmap * 255]]


def run_file(fileName):
    frame = cv2.imread(fileName)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # [trans, trans_refined, radiance, dark, depthmap] = perform(frame)
    return haze_rm(frame)


fileName = np.random.choice(glob.glob('./input/train/*.jpg'), 1)[0]
tr = pd.read_csv('./input/train_v2.csv')
labels = tr.loc[tr['image_name'] == fileName.split('\\')[-1].split('.')[0],
                'tags'].values[0]
[trans, trans_refined, radiance, dark, depthmap] = run_file(fileName)

fig, ax = plt.subplots(3, 2, figsize=(6, 9))
ax = ax.ravel()
ax[0].imshow(cv2.cvtColor(cv2.imread(fileName), cv2.COLOR_BGR2RGB))
ax[0].set_title('Original: \n{}'.format(labels), color='slategray')
ax[1].imshow(radiance)
ax[1].set_title('Haze Removed', color='slategray')
ax[2].imshow(dark, cmap='RdBu_r')
ax[2].set_title('Dark Prior Channel', color='slategray')
ax[3].imshow(trans, cmap='RdBu_r')
ax[3].set_title('Transparency', color='slategray')
ax[4].imshow(depthmap, cmap='RdBu_r')
ax[4].set_title('Depth Map', color='slategray')
ax[5].imshow(trans_refined, cmap='RdBu_r')
ax[5].set_title('Refined Transparency', color='slategray')
fig.tight_layout()

fig.savefig('./fig/' + fileName.split('\\')[-1].split('.')[0] + '.png',
            dpi=140)
