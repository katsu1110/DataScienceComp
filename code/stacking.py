import os
import sys

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, log_loss, roc_auc_score

# stacking
class Stacking(object):
    """
    perform stacking ensemble
    - oof, ypred are (samples x models) numpy array
    - yval (samples, ) numpy array 
    - n_splits : K in KFold (default is 100)
    - seed : seed (int)
    - objective : regression, multiclass, or binary
    """

    def __init__(self, oof, yval, ypred, n_splits=100, seed=1220, task="regression"):
        # objective can be either "regression" or "classification"
        self.oof = oof # out-of-fold prediction
        self.yval = yval # actual target in the out-of-fold
        self.ypred = ypred # prediction for test
        self.n_splits = n_splits
        self.seed = seed
        self.task = task

    def calc_metric(self, y_true, y_pred): # this may need to be changed based on the metric of interest
        if self.task == "multiclass":
            return log_loss(y_true, y_pred)
        elif self.task == "binary":
            return roc_auc_score(y_true, y_pred)
        elif self.task == "regression":
            return np.sqrt(mean_squared_error(y_true, y_pred))

    def fit(self):
        # z-scoring
        scaler = StandardScaler()
        self.oof = scaler.fit_transform(self.oof)
        self.ypred = scaler.transform(self.ypred)

        # train a linear model
        kf = KFold(n_splits=self.n_splits, random_state=self.seed, shuffle=True)
        models = []
        scores = []
        for fold, (train_index, test_index) in enumerate(kf.split(self.oof)):
            # train test split
            X_train, X_test = self.oof[train_index, :], self.oof[test_index, :]
            y_train, y_test = self.yval[train_index], self.yval[test_index]

            # model
            if self.task == "regression":
                mdl = linear_model.Ridge(**{'alpha': 220, 'solver': 'lsqr', 'fit_intercept': True,
                                      'max_iter': 5000, 'random_state': self.seed * (fold + 1)})
                mdl.fit(X_train, y_train)
                preds = mdl.predict(X_test)
            elif (self.task == "binary") | (self.task == "multiclass"):
                mdl = linear_model.LogisticRegression(penalty='l2', tol=0.0001, C=1.0, fit_intercept=True, 
                                        random_state=self.seed * (fold + 1), solver='lbfgs', max_iter=5000, 
                                        multi_class='auto', verbose=0, warm_start=False)
                mdl.fit(X_train, y_train)
                preds = mdl.predict_proba(X_test)
                if self.task == "binary":
                    preds = preds[:, 1]
                elif self.task == "multiclass":
                    preds = np.argmax(preds, axis=1)
            
            # store
            models.append(mdl)
            score = self.calc_metric(y_test, preds)
            scores.append(score)
            print(score)

        # predict
        final_pred = np.zeros(self.ypred.shape[0])
        for m in models:
            final_pred += m.predict(self.ypred)

        return final_pred / len(models)
