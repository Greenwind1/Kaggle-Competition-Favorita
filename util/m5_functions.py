import sys

sys.path.append('..')
import warnings
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


def reduce_mem_num(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            # c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.uint32)
                elif c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    if verbose: print(
        'Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem)
    )

    return df
