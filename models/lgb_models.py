import numpy as np
import pandas as pd
import lightgbm as lgb

from base_models import BaseModel

class LgbModel(BaseModel):
    """
    LGB wrapper

    """

    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        model = lgb.train(self.params, train_set, num_boost_round = 3000, valid_sets=[train_set, val_set], verbose_eval=verbosity)
        fi = model.feature_importance(importance_type="gain")
        return model, fi

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)
        val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)
        return train_set, val_set

    def get_params(self):
        params = {
                    'n_estimators': 1024,
                    'boosting_type': 'gbdt',
                    'subsample': 0.75,
                    'subsample_freq': 1,
                    'learning_rate': 0.07,
                    'feature_fraction': 0.9,
                    'max_depth': 15,
                    'lambda_l1': 1,
                    'lambda_l2': 1,
                    'early_stopping_rounds': 100
                    }
        if self.task == "regression":
            params["objective"] = "regression"
            params["metric"] = "rmse"
        elif self.task == "classification":
            params["objective"] = "binary"
            params["metric"] = "auc"

        return params
