import numpy as np
from sklearn import linear_model
import optuna

def lin_model(cls, train_set, val_set):
    """
    Linear model hyperparameters and models
    """

    params = {
            'max_iter': 8000,
            'fit_intercept': True,
            'random_state': cls.seed
        }

    if cls.task == "regression":
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
        model = linear_model.Ridge(**{'alpha': 220, 'solver': 'lsqr', 'fit_intercept': params['fit_intercept'],
                                'max_iter': params['max_iter'], 'random_state': params['random_state']})
    elif (cls.task == "binary") | (cls.task == "multiclass"):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        model = linear_model.LogisticRegression(**{"C": 1.0, "fit_intercept": params['fit_intercept'], 
                                "random_state": params['random_state'], "solver": "lbfgs", "max_iter": params['max_iter'], 
                                "multi_class": 'auto', "verbose":0, "warm_start":False})
                                
    model.fit(train_set['X'], train_set['y'])

    # feature importance (for multitask, absolute value is computed for each feature)
    if cls.task == "multiclass":
        fi = np.mean(np.abs(model.coef_), axis=0).ravel()
    else:
        fi = model.coef_.ravel()

    return model, fi
