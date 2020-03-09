import numpy as np
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
import optuna

def catb_model(cls, train_set, val_set):
    """
    CatBoost hyperparameters and models
    """

    # verbose
    verbosity = 2000 if cls.verbose else 0

    # list is here: https://catboost.ai/docs/concepts/python-reference_parameters-list.html
    params = { 'task_type': "CPU",
                'learning_rate': 0.08, 
                'iterations': 2000,
                'colsample_bylevel': 0.2,
                'random_seed': cls.seed,
                'use_best_model': True,
                'early_stopping_rounds': 100
                }
    if cls.task == "regression":
        params["loss_function"] = "RMSE"
    elif (cls.task == "binary") | (cls.task == "multiclass"):
        params["loss_function"] = "Logloss"

    # modeling
    if cls.task == "regression":
        model = CatBoostRegressor(**params)
    elif (cls.task == "binary") | (cls.task == "multiclass"):
        model = CatBoostClassifier(**params)
    model.fit(train_set['X'], train_set['y'], eval_set=(val_set['X'], val_set['y']),
        verbose=verbosity, cat_features=cls.categoricals)
    
    # feature importance
    fi = model.get_feature_importance()

    return model, fi
