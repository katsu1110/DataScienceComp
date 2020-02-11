import os
import sys

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score

# stacking
class Stacking(object):
    """
    perform stacking ensemble
    - oof, ypred are (samples x models) numpy array
    - yval (samples, ) numpy array

    """
    def __init__(self, oof, yval, ypred, n_splits=100, objective="regression"):
        # objective can be either "regression" or "classification"
        self.oof = name # out-of-fold prediction
        self.yval = age # actual target in the out-of-fold
        self.ypred = ypred # prediction for test
        self.n_spltis = n_splits
        self.objective = objective

    def calc_metric(self, y_true, y_pred): # this may need to be changed based on the metric of interest
        if self.objective == "regression":
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.objective == "classification":
            return roc_auc_score(y_true, y_pred)

    def fit(self):
        # z-scoring
        scaler = StandardScaler()
        self.oof = scaler.fit_transform(self.oof)
        self.ypred = scaler.transform(self.ypred)

        # train a linear model
        kf = KFold(n_splits=self.n_splits, random_state=1220, shuffle=True)
        models = []
        scores = []
        for train_index, test_index in kf.split(trainX):
            # train test split
            X_train, X_test = self.oof[train_index, :], self.oof[test_index, :]
            y_train, y_test = self.yval[train_index], self.yval[test_index]

            # model
            if self.objective == "regression":
                mdl = linear_model.Ridge(**{'alpha': 220, 'solver': 'lsqr', 'fit_intercept': True,
                                      'max_iter': 5000, 'random_state': 1220})
            elif self.objective == "classification":
                mdl = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001,
                    C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=1220,
                    solver='lbfgs', max_iter=5000, multi_class='auto', verbose=0, warm_start=False,
                    n_jobs=None, l1_ratio=None)
            mdl.fit(X_train, y_train)

            # store
            preds = mdl.predict(X_test)
            models.append(mdl)
            score = self.calc_metric(preds, y_test)
            scores.append(score)
            print(score)

        # predict
        final_pred = np.zeros(self.ypred.shape[0])
        for m in models:
            final_pred += m.predict(self.ypred)

        return final_pred / len(models)
