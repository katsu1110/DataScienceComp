import numpy as np
import pandas as pd
from sklearn import utils
from catboost import CatBoostRegressor, CatBoostClassifier

def catb_model(cls, train_set, val_set):
    """
    CatBoost hyperparameters and models
    """

    # verbose
    verbosity = 100 if cls.verbose else 0

    # list is here: https://catboost.ai/docs/concepts/python-reference_parameters-list.html
    if not cls.params:
        params = { 'task_type': "CPU",
                    'learning_rate': 0.08, 
                    'iterations': 24000,
                    'colsample_bylevel': 0.4,
                    'random_seed': cls.seed,
                    'use_best_model': True,
                    'early_stopping_rounds': 80
                    }
        if cls.task == "regression":
            params["loss_function"] = "RMSE"
            params["eval_metric"] = "RMSE"
        elif cls.task == "binary":
            params["loss_function"] = "Logloss"
            params["eval_metric"] = "AUC"
        elif cls.task == "multiclass":
            params["loss_function"] = "MultiClass"
            params["eval_metric"] = "MultiClass"
        cls.params = params

    # modeling
    if cls.task == "regression":
        model = CatBoostRegressor(**cls.params)
    elif (cls.task == "binary") | (cls.task == "multiclass"):
        cw = utils.class_weight.compute_class_weight('balanced', np.unique(train_set['y']), train_set['y'])
        cls.params['class_weights'] = cw
        model = CatBoostClassifier(**cls.params)
    model.fit(train_set['X'], train_set['y'], eval_set=(val_set['X'], val_set['y']),
        verbose=verbosity, cat_features=cls.categoricals)
    
    # feature importance
    fi = model.get_feature_importance()

    return model, fi
