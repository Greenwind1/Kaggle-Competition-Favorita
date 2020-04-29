# -*- coding: utf-8 -*-

import os
import pandas as pd

os.getcwd()

# change root directory if necessary,
# and reboot python console.

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)
# dir(pd.options.display)


os.makedirs('./input', exist_ok=True)
# os.makedirs('./input/raw', exist_ok=True)
# os.makedirs('./data', exist_ok=True)
os.makedirs('./code', exist_ok=True)
os.makedirs('./code/old', exist_ok=True)
os.makedirs('./features', exist_ok=True)
os.makedirs('./fig', exist_ok=True)
os.makedirs('./model', exist_ok=True)
os.makedirs('./util', exist_ok=True)
os.makedirs('./fold', exist_ok=True)
os.makedirs('./ref', exist_ok=True)
os.makedirs('./log', exist_ok=True)
os.makedirs('./old', exist_ok=True)
# os.makedirs('./xgbfir', exist_ok=True)


os.makedirs('./oof', exist_ok=True)
os.makedirs('./oof/dnn', exist_ok=True)
os.makedirs('./oof/ext', exist_ok=True)
os.makedirs('./oof/lgb', exist_ok=True)
os.makedirs('./oof/mf', exist_ok=True)
os.makedirs('./oof/logit', exist_ok=True)
os.makedirs('./oof/rf', exist_ok=True)
os.makedirs('./oof/xgb', exist_ok=True)
os.makedirs('./oof/svm', exist_ok=True)
os.makedirs('./oof/knn', exist_ok=True)
os.makedirs('./oof/cb', exist_ok=True)
os.makedirs('./oof/kernel', exist_ok=True)
os.makedirs('./oof/blend', exist_ok=True)
os.makedirs('./oof/stack', exist_ok=True)

os.makedirs('./sub', exist_ok=True)
os.makedirs('./sub/dnn', exist_ok=True)
os.makedirs('./sub/ext', exist_ok=True)
os.makedirs('./sub/lgb', exist_ok=True)
os.makedirs('./sub/mf', exist_ok=True)
os.makedirs('./sub/logit', exist_ok=True)
os.makedirs('./sub/rf', exist_ok=True)
os.makedirs('./sub/xgb', exist_ok=True)
os.makedirs('./sub/svm', exist_ok=True)
os.makedirs('./sub/knn', exist_ok=True)
os.makedirs('./sub/cb', exist_ok=True)
# os.makedirs('./sub/kernel', exist_ok=True)
os.makedirs('./sub/blend', exist_ok=True)
os.makedirs('./sub/stack', exist_ok=True)
