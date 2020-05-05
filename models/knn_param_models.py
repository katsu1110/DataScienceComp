import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

def knn_model(cls, train_set, val_set):
    """
    kNN model hyperparameters and models
    """

    params = {
            'n_neighbors': 10,
            'weights': 'distance',
            'algorithm': 'auto',
            'p': 1
        }

    if cls.task == "regression":
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
        model = KNeighborsRegressor(**params)
    elif (cls.task == "binary") | (cls.task == "multiclass"):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        model = KNeighborsClassifier(**params)
                                
    model.fit(train_set['X'], train_set['y'])

    # permutation importance to get a feature importance (off in default)
    # fi = PermulationImportance(model, train_set['X'], train_set['y'], cls.features)
    fi = np.zeros(len(cls.features)) # no feature importance computed

    return model, fi
