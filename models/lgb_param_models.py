import numpy as np
import pandas as pd
import lightgbm as lgb

def lgb_model(cls, train_set, val_set):
    """
    LightGBM hyperparameters and models
    """

    # verbose
    verbosity = 500 if cls.verbose else 0

    # list is here: https://lightgbm.readthedocs.io/en/latest/Parameters.html
    params = {
                'n_estimators': 4000,
                'objective': cls.task,
                'boosting_type': 'gbdt',
                'num_leaves': 128,
                'min_data_in_leaf': 72,
                'max_depth': -1,
                'learning_rate': 0.08,
                'subsample': 0.76,
                'subsample_freq': 1,
                'feature_fraction': 0.2,
                'lambda_l1': 1,
                'lambda_l2': 1,
                'seed': cls.seed,
                'early_stopping_rounds': 80,
                'num_boost_round': 2000
                }    
    if cls.task == "regression":
        params["metric"] = "rmse"
    elif cls.task == "binary":
        params["metric"] = "auc" # other candidates: binary_logloss
        params["is_unbalance"] = True # assume unbalanced data
    elif cls.task == "multiclass":
        params["metric"] = "multi_logloss" # other candidates: cross_entropy, auc_mu
        params["num_class"] = len(np.unique(cls.train_df[cls.target].values))
        params["class_weight"] = 'balanced' # assume unbalanced data

    # modeling and feature importance
    if cls.task == "multiclass": # sklearn API for 'class_weight' implementation
        model = lgb.LGBMClassifier(**params)
        model.fit(train_set['X'], train_set['y'], eval_set=[(val_set['X'], val_set['y'])],
            verbose=verbosity, categorical_feature=cls.categoricals)
        fi = model.booster_.feature_importance(importance_type="gain")
    else: # python API for efficient memory usage
        model = lgb.train(params, train_set, valid_sets=[train_set, val_set], verbose_eval=verbosity)
        fi = model.feature_importance(importance_type="gain")

    return model, fi
