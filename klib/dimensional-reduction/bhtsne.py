# -*- coding: utf-8 -*-

import numpy as np
import sklearn.base
import bhtsne


class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1):
        self.dimensions = dimensions
        self.perplexity = perplexity
        self.theta = theta
        self.rand_seed = rand_seed

    def fit_transform(self, x, **kwargs):
        return bhtsne.tsne(x.astype(np.float64),
                           dimensions=self.dimensions,
                           perplexity=self.perplexity,
                           theta=self.theta,
                           rand_seed=self.rand_seed)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler

    cancer = load_breast_cancer()

    ss = StandardScaler()

    tsne = BHTSNE(dimensions=2, perplexity=30, theta=0.5, rand_seed=-1)
    cancer_tsne = tsne.fit_transform(ss.fit_transform(cancer.data))
    cancer_tsne = tsne.fit_transform(cancer.data)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(cancer_tsne[cancer.target == 1, 0],
            cancer_tsne[cancer.target == 1, 1],
            '.', markersize=5,
            color='deeppink', alpha=0.7,
            label='positive')
    ax.plot(cancer_tsne[cancer.target == 0, 0],
            cancer_tsne[cancer.target == 0, 1],
            '.', markersize=5,
            color='darkslateblue', alpha=0.7,
            label='negative')
    fig.legend()
    fig.tight_layout()
    fig.show()
