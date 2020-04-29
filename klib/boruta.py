# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

plt.style.use('ggplot')
warnings.simplefilter(action='ignore', category=FutureWarning)

CPU = psutil.cpu_count() - 1
SEED = 71
boston = load_boston()
train_x, train_y = boston.data, boston.target

# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestRegressor(n_jobs=CPU,
                           max_depth=5)

# define Boruta feature selection method
bor = BorutaPy(estimator=rf,
               n_estimators=1000,
               perc=100,
               alpha=0.05,
               two_step=False,
               max_iter=100,
               random_state=2019,
               verbose=1)
bor.fit(train_x, train_y)

br_df = pd.DataFrame({'f_name': boston.feature_names})
br_df['importance'] = bor._get_imp(train_x, train_y)
br_df['ranking'] = bor.ranking_
br_df['support'] = bor.support_
br_df.sort_values(by='importance', ascending=False, inplace=True)

fig, ax = plt.subplots(figsize=(5, 5))
ax.barh(range(len(br_df)), br_df['importance'],
        color='deeppink', align='center', height=0.5, alpha=0.8)
ax.set_yticks(range(len(br_df)))
ax.set_yticklabels(br_df['f_name'] + '(' + br_df['support'].astype(str) + ')',
                   fontsize=10)
ax.invert_yaxis()
fig.tight_layout()
fig.show()

train_x_sel = bor.transform(train_x)
