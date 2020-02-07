import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from base_classifier import ClassifierBase

class CatbClassifier(ClassifierBase):
    """
    CatBoost regressor wrapper

    """
    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        clf = CatBoostRegressor(**self.params)
        clf.fit(train_set['X'], train_set['y'], eval_set=(val_set['X'], val_set['y']),
            verbose=verbosity, cat_features=self.categoricals)
        return clf, clf.get_feature_importance()

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = {'X': x_train, 'y': y_train}
        val_set = {'X': x_val, 'y': y_val}
        return train_set, val_set

    def get_params(self):
        params = {'loss_function': 'Logloss',
                   'task_type': "CPU",
                   'iterations': 2000,
                   'od_type': "Iter",
                   'depth': 10,
                   'colsample_bylevel': 0.5,
                   'early_stopping_rounds': 300,
                   'l2_leaf_reg': 18,
                   'random_seed': 42,
                   'use_best_model': True
                    }
        return params
