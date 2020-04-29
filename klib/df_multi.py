import pandas as pd
import numpy as np
from multiprocessing import Pool

df = pd.DataFrame(dict(animal=['monkey', 'dog', 'monkey', 'cat', 'dog'],
                       count=[2, 4, 1, 4, 5]))


# row only
def parallel_df_row(df, num_cpu, map_func):
    p = Pool(num_cpu)
    df_split = np.array_split(df, num_cpu)  # return list type object
    result = p.map(map_func, df_split)
    p.close()
    return pd.concat(result)


def map_func1(df):
    df['count'] *= 3
    return df


# column and row
def parallel_df(df, func, columnwise=False, num_cpu=2):
    num_partitions = num_cpu
    num_cores = num_cpu
    p = Pool(num_cores)

    if columnwise:
        # column parallelization
        df_split = [df[col_name] for col_name in df.columns]
        df = pd.concat(p.map(func, df_split), axis=1)
    else:
        # row parallelization
        df_split = np.array_split(df, num_partitions)
        df = pd.concat(p.map(func, df_split))

    p.close()
    p.join()
    return df


def map_func2(series: pd.Series):
    return series


if __name__ == '__main__':
    df = parallel_df_row(df, 2, map_func1)
    print(df)
    df = parallel_df(df, map_func2, columnwise=True)
    print(df)
