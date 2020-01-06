import os

import numpy as np
import pandas as pd
import xgboost as xgb

from base import Base_Model

class Xgb_Model(Base_Model):

    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        bst = xgb.train(self.params, train_set,
                         num_boost_round=5000, evals=[(train_set, 'train'), (val_set, 'val')],
                         verbose_eval=verbosity, early_stopping_rounds=100)
        fi = np.zeros(len(self.features))
        xgbd = bst.get_score(importance_type='gain')
        for i, f in enumerate(self.features):
            try:
                fi[i] = xgbd[f]
            except:
                pass
        return bst, fi

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = xgb.DMatrix(x_train, y_train)
        val_set = xgb.DMatrix(x_val, y_val)
        return train_set, val_set

    def convert_x(self, x):
        return xgb.DMatrix(x)

    def get_params(self):
        params = {'colsample_bytree': 0.8,
            'learning_rate': 0.01,
            'max_depth': 10,
            'subsample': 1,
            'objective':'reg:squarederror',
            #'eval_metric':'rmse',
            'min_child_weight':3,
            'gamma':0.25,
            'n_estimators':5000}

        return params
