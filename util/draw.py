# -*- coding: utf-8 -*-

import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')


def draw1(oof_actuals: pd.DataFrame, oof_preds: pd.DataFrame,
          oof_rmsse: pd.Series, x_val_date: list, fname: str,
          keys=['all', 'CA', 'TX', 'WI'],
          figsize=(10, 10)):
    fig, ax = plt.subplots(len(keys), 1, figsize=figsize)
    ax = ax.flatten()
    for i, k in enumerate(keys):
        ax[i].plot(x_val_date, oof_actuals.loc[k, :], label='actual',
                   marker='.', ms=4, linewidth=1, color='gray')
        ax[i].plot(x_val_date, oof_preds.loc[k, :], label='pred',
                   marker='.', ms=4, linewidth=1, color='deeppink')
        ax[i].tick_params(axis='x', labelsize=5, color='gray')
        ax[i].set_title(k + f' (RMSSE = {oof_rmsse[k]:.5f})',
                        color='gray', fontsize=10)
        if i == 0: ax[i].legend(loc='lower right', framealpha=0, fontsize=7)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=100)
    plt.close()
