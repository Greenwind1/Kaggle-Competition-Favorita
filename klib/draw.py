# -*- coding: utf-8 -*-

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass
    print(cm)

    tick_marks = np.arange(len(classes))
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va='center',
                 color="white" if cm[i, j] > thresh else "black")
    ax.grid(linewidth=0.2, color='snow', alpha=0.2)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_xbound(np.min(tick_marks) - 0.5, np.max(tick_marks) + 0.5)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.set_ybound(np.min(tick_marks) - 0.5, np.max(tick_marks) + 0.5)
    ax.set_title(title, color='dimgray', size=10)
    ax.set_xlabel('Predicted', color='dimgray', fontsize=8)
    ax.set_ylabel('Actual', color='dimgray', fontsize=8)


def plot_multilabel_fbeta(cm, label, beta=1, score=0):
    """ plot a bar graph of fbeta score before class averaged
    :arg cm: multi-labeled confusion matrix array
    - - -
    fbeta = (1 + beta^2) * TP / ( (1 + beta^2) * TP + beta^2 * FN + FP )
    """
    fbeta_l = [(1 + beta ** 2) * m[1, 1] /
               ((1 + beta ** 2) * m[1, 1] + beta ** 2 * m[1, 0] + m[0, 1])
               for m in cm]
    macro_f1 = np.round(np.array(fbeta_l).mean(), 4)
    score = np.round(score, 4)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.barh(label, fbeta_l, height=0.7, color='slategray')
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(0.2)
    ax.set_title('macro_f1: {}, mean_f1: {}'.format(macro_f1, score),
                 color='slategray', size=10)
    fig.tight_layout()


def plot_learning_history(log, lr=True):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(log['epoch'], log['loss'], '.', label='tr-loss')
    ax.plot(log['epoch'], log['val_loss'], '.', label='val-loss')
    if lr:
        twin_ax = ax.twinx()
        twin_ax.plot(log['epoch'], log['lr'],
                     label='lr', color='slategray')
    ax.legend()
    ax.grid(0.2)
    ax.set_title('Learning History', color='slategray', size=10)
    fig.tight_layout()
