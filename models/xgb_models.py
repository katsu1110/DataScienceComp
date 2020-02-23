import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from optuna.visualization import plot_optimization_history
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error
from base_models import BaseModel

class XgbModel(BaseModel):
    """
    XGB wrapper

    """

    def train_model(self, train_set, val_set):
        verbosity = 1000 if self.verbose else 0
        model = xgb.train(self.params, train_set, 
                         num_boost_round=2000, evals=[(train_set, 'train'), (val_set, 'val')], 
                         verbose_eval=verbosity, early_stopping_rounds=100)
        return model, np.asarray(list(model.get_score(importance_type='gain').values()))
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = xgb.DMatrix(x_train, y_train)
        val_set = xgb.DMatrix(x_val, y_val)
        return train_set, val_set
    
    def convert_x(self, x):
        return xgb.DMatrix(x)
        
    def get_params(self):
        params = {
            'colsample_bytree': 0.8,                 
            'learning_rate': 0.05,
            'max_depth': 10,
            'subsample': 1,
            'min_child_weight':3,
            'gamma':0.25,
            'seed': self.seed,
            'n_estimators':4000
            }

        # list is here: https://xgboost.readthedocs.io/en/latest/parameter.html
        if self.task == "regression":
            params["objective"] = 'reg:squarederror'
            params["eval_metric"] = "rmse"
        elif self.task == "binary":
            params["objective"] = 'binary:logistic'
            params["eval_metric"] = "auc"
        elif self.task == "multiclass":
            params["objective"] = 'multi:softmax'
            params["eval_metric"] = "mlogloss" 

        return params