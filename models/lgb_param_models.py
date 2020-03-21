import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

def lgb_model(cls, train_set, val_set):
    """
    LightGBM hyperparameters and models
    """

    # verbose
    verbosity = 2000 if cls.verbose else 0

    # list is here: https://lightgbm.readthedocs.io/en/latest/Parameters.html
    params = {
                'n_estimators': 4000,
                'objective': cls.task,
                'boosting_type': 'gbdt',
                'num_leaves': 128,
                'min_data_in_leaf': 64,
                'max_depth': -1,
                'learning_rate': 0.04,
                'subsample': 0.76,
                'subsample_freq': 1,
                'feature_fraction': 0.2,
                'seed': cls.seed,
                'early_stopping_rounds': 100
                }    
    if cls.task == "regression":
        params["metric"] = "rmse"
    elif cls.task == "binary":
        params["metric"] = "auc" # binary_logloss
    elif cls.task == "multiclass":
        params["metric"] = "multi_logloss" # cross_entropy, auc_mu
        params["num_class"] = len(np.unique(cls.train_df[cls.target].values))

    # modeling
    model = lgb.train(params, train_set, num_boost_round=2000, valid_sets=[train_set, val_set], verbose_eval=verbosity)
            
    # feature importance
    fi = model.feature_importance(importance_type="gain")

    return model, fi
