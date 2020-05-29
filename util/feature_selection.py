import sys, warnings

sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')


# from tensorflow.python.util import deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False

def col_remove(df: pd.DataFrame, col_l: list):
    ncol = df.shape[1]
    if col_l:
        print(col_l)
        print(f'{len(col_l)} columns removed from {ncol} columns.')
        return df.drop(col_l, axis=1)
    else:
        print(f'No column removed from {ncol} columns.')
        return df


def nan_list(df: pd.DataFrame, th_nan: np.float):
    col_rm_l = df.columns[df.notnull().sum() / len(df) < th_nan].tolist()
    if len(col_rm_l) > 0:
        print(f'NaN of {len(col_rm_l)} columns are under {th_nan:.5f}')
        return col_rm_l


def low_cov_list(df: pd.DataFrame, th_cov: np.float):
    tmp_cov = df.std(axis=0) / df.mean(axis=0)
    col_rm_l = tmp_cov.index[tmp_cov <= th_cov].tolist()
    if len(col_rm_l) > 0:
        print(f'CoV of {len(col_rm_l)} columns are under {th_cov:.5f}')
        return col_rm_l


def cor_list(df: pd.DataFrame, th: np.float, method='spearman'):
    df_cor = df.corr(method=method)
    df_cor = abs(df_cor)
    columns = df_cor.columns

    del_col = []

    # set diagonal elements to zero
    for i in range(0, len(columns)):
        df_cor.iloc[i, i] = 0

    while True:
        max_col_cor_vals = df_cor.max()
        max_cor = max_col_cor_vals.max()
        query_col = max_col_cor_vals.idxmax()
        target_col = df_cor[query_col].idxmax()

        if max_cor < th:
            break
        else:
            # if sum of target_col correlation values with other columns
            # is larger than the sum of query_col, delete target_col.
            if sum(df_cor[query_col]) <= sum(df_cor[target_col]):
                del_col.append(target_col)
                df_cor.drop([target_col], axis=0, inplace=True)
                df_cor.drop([target_col], axis=1, inplace=True)
            else:
                del_col.append(query_col)
                df_cor.drop([query_col], axis=0, inplace=True)
                df_cor.drop([query_col], axis=1, inplace=True)

    return del_col
