# -*- coding: utf-8 -*-

import psutil
from joblib import Parallel, delayed
from time import time

CPU = psutil.cpu_count() - 1


def process(n):
    return sum([i * n for i in range(100000)])


start = time()

""" JOBLIB USAGE """
r = Parallel(n_jobs=CPU, verbose=1)(
    # generator
    (delayed(process)(i) for i in range(10000)))
print(sum(r))

print('{}(s) elapsed.'.format(time() - start))
