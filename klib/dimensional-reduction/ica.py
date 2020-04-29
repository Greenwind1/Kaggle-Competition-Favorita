# -*- coding: utf-8 -*-

import gc
import os
import warnings
import psutil
import json
import pickle
import collections as cl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from sklearn.decomposition import FastICA, PCA

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None
# dir(pd.options.display)
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')

"""
ENVIROMENT VARIABLES
"""
CODE_N = 'ica'

CPU = psutil.cpu_count() - 1
SEED = 71
np.random.seed(SEED)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# ESL p641, S_ is derived from SVD
# X = UDV^T and U is an orthogonal matrix.
# Though depending on definition, S_ is proportional to U.
for s in range(S_.shape[1]):
    print('component{} of S_ : mean {:.6f}, var {:.6f}'.format(
        s, S_[:, s].mean(), S_[:, s].var() * S_.shape[0]))

# For comparison, compute PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

# Plot results
models = [X, S, S_, H]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['deeppink', 'darkslateblue', 'limegreen']

fig, ax = plt.subplots(4, 1, figsize=(8, 6))
axes = ax.ravel()
for i, (model, name) in enumerate(zip(models, names), 1):
    for sig, color in zip(model.T, colors):
        axes[i - 1].plot(sig, '-',
                         color=color,
                         linewidth=1,
                         ms=1,
                         alpha=0.5)
    axes[i - 1].set_title(name)
fig.tight_layout()
fig.show()
