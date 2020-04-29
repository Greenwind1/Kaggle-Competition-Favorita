# -*- coding: utf-8 -*-

import gc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import eli5

plt.style.use('ggplot')


# SIMPLE LGBM IMPORTANCE
def lgbm_plot_importance(lgbm, split_csv_name, gain_csv_name, png_name):
    fi_df_s = pd.DataFrame({
        'name': lgbm.feature_name(),
        'FI': lgbm.feature_importance(importance_type='split'),
    }).sort_values('FI', ascending=False)
    fi_df_s.to_csv(split_csv_name)
    fi_df_g = pd.DataFrame({
        'name': lgbm.feature_name(),
        'FI': lgbm.feature_importance(importance_type='gain'),
    }).sort_values('FI', ascending=False)
    fi_df_g.to_csv(gain_csv_name)
    if len(fi_df_s) >= 100:
        plot_n = 100
    else:
        plot_n = len(fi_df_s)
    fig, ax = plt.subplots(1, 2, figsize=[18, 10])
    ax[0].barh(range(plot_n), fi_df_s['FI'][:plot_n],
               height=0.5, color='deeppink')
    ax[0].set_yticks(range(plot_n))
    ax[0].set_yticklabels(fi_df_s['name'][:plot_n], fontsize=6)
    ax[0].invert_yaxis()
    ax[0].set_title('split importance')
    ax[1].barh(range(plot_n), fi_df_g['FI'][:plot_n],
               height=0.5, color='limegreen')
    ax[1].set_yticks(range(plot_n))
    ax[1].set_yticklabels(fi_df_s['name'][:plot_n], fontsize=6)
    ax[1].invert_yaxis()
    ax[1].set_title('gain importance')
    fig.tight_layout()
    fig.show()
    fig.savefig(png_name, dpi=140)
    return fi_df_s, fi_df_g


# -------------------------------------------------------------------
#   Simple XGBoost Importance Plotter
# -------------------------------------------------------------------
def xgbm_plot_importance(xgbt, split_csv_name, gain_csv_name, png_name):
    """
    plot xgbm feature importance
    :param xgbt: xgboost model object (data should have feature names)
    :param split_csv_name:
    :param gain_csv_name:
    :param png_name:
    :return: feature importance split/gain dataframes
    """
    fi_df_s = pd.Series(
        data=xgbt.get_score(importance_type='weight')
    ).to_frame().reset_index()
    fi_df_s.columns = ['name', 'FI']
    fi_df_s = fi_df_s.sort_values('FI', ascending=False)
    fi_df_s.to_csv(split_csv_name)

    fi_df_g = pd.Series(
        xgbt.get_score(importance_type='gain')
    ).to_frame().reset_index()
    fi_df_g.columns = ['name', 'FI']
    fi_df_g = fi_df_g.sort_values('FI', ascending=False)
    fi_df_g.to_csv(gain_csv_name)

    if len(fi_df_s) >= 100:
        plot_n = 100
    else:
        plot_n = len(fi_df_s)

    fig, ax = plt.subplots(1, 2, figsize=[18, 10])
    ax[0].barh(range(plot_n), fi_df_s['FI'][:plot_n],
               height=0.5, color='deeppink')
    ax[0].set_yticks(range(plot_n))
    ax[0].set_yticklabels(fi_df_s['name'][:plot_n], fontsize=6)
    ax[0].invert_yaxis()
    ax[0].set_title('split importance')
    ax[1].barh(range(plot_n), fi_df_g['FI'][:plot_n],
               height=0.5, color='limegreen')
    ax[1].set_yticks(range(plot_n))
    ax[1].set_yticklabels(fi_df_s['name'][:plot_n], fontsize=6)
    ax[1].invert_yaxis()
    ax[1].set_title('average gain importance')
    fig.tight_layout()
    fig.show()
    fig.savefig(png_name, dpi=140)
    plt.close()

    return fi_df_s, fi_df_g


# SIMPLE EXTRA TREE IMPORTANCE
def ext_plot_importance(ext, name, csv_name, png_name):
    importances = ext.feature_importances_
    std = np.std([t.feature_importances_ for t in ext.estimators_],
                 axis=0)
    fi_df = pd.Series(importances).to_frame()
    fi_df.columns = ['FI']
    fi_df['name'] = name
    fi_df['FI_STD'] = std
    fi_df = fi_df.sort_values('FI', ascending=False)
    fi_df.to_csv(csv_name)
    if len(fi_df) >= 100:
        plot_n = 100
    else:
        plot_n = len(fi_df)
    fig, ax = plt.subplots(1, 1, figsize=[9, 10])
    ax.barh(range(100), fi_df['FI'][:100], height=0.5, color='deeppink')
    ax.set_yticks(range(100))
    ax.set_yticklabels(fi_df['name'][:100], fontsize=6)
    ax.invert_yaxis()
    ax.set_title('importance')
    fig.tight_layout()
    fig.show()
    fig.savefig(png_name, dpi=140)
    return fi_df


# PERMUTATION IMPORTANCE
def lgbm_eli_imp(lgb, imp_csv_name, png_name):
    return 0
