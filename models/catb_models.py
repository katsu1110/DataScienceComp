import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier

from base_models import BaseModel

class CatbModel(BaseModel):
    """
    CatBoost wrapper

    """
    def train_model(self, train_set, val_set):
        verbosity = 1000 if self.verbose else 0
        if self.task == "regression":
            model = CatBoostRegressor(**self.params)
        elif (self.task == "binary") | (self.task == "multiclass"):
            model = CatBoostClassifier(**self.params)
        model.fit(train_set['X'], train_set['y'], eval_set=(val_set['X'], val_set['y']),
            verbose=verbosity, cat_features=self.categoricals)
        return model, model.get_feature_importance()

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = {'X': x_train, 'y': y_train}
        val_set = {'X': x_val, 'y': y_val}
        return train_set, val_set

    def convert_x(self, x):
        return x
        
    def get_params(self):
        params = { 'task_type': "CPU",
                   'learning_rate': 0.03, 
                   'iterations': 5000,
                   'random_seed': self.seed,
                   'use_best_model': True,
                   'early_stopping_rounds': 100
                    }
        if self.task == "regression":
            params["loss_function"] = "RMSE"
        elif (self.task == "binary") | (self.task == "multiclass"):
            params["loss_function"] = "Logloss"
        return params
