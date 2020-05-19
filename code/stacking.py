import os
import sys
import numpy as np
import pandas as pd
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, f1_score

# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
sns.set_context("talk")
style.use('ggplot')

# stacking
class Stacking(object):
    """
    perform stacking ensemble

    :INPUTS:

    :oof: dictionary {ModelName : oof (samples x models) numpy array, ...}. Note that its key has to match ypred
    :yval: (samples, ) numpy array
    :ypred: dictionary {ModelName : ypred (samples x models) numpy array, ...}. Note that its key has to match oof
    :n_splits: K in KFold (default is 100)
    :seed: seed (int)
    :task: regression, multiclass, or binary

    :EXAMPLE:

    # load data
    oof = {'lgb' : np.load("../input/lgb/oof.npy"), 'mlp' : np.load("../input/mlp/oof.npy")}
    yval = train[target].values
    ypred = {'lgb' : np.load("../input/lgb/ypred.npy"), 'mlp' : np.load("../input/mlp/ypred.npy")}

    # run stacking ensemble
    s = Stacking(oof, yval, ypred, task="regression")
    stacking_oof, stacking_pred = s.fit()
    """

    def __init__(self, oof : Dict, yval : np.ndarray, ypred : Dict, n_splits : int=100, seed : int=1220, task : str="regression"):
        self.oof = oof
        self.yval = yval 
        self.ypred = ypred 
        self.n_splits = n_splits
        self.seed = seed
        self.task = task

    def data_formatter(self):
        """
        format dict data into pandas dataframe
        """    

        for i, k in enumerate(self.oof.keys()):
            if i == 0:
                train = pd.DataFrame(data=self.oof[k], columns=[f"{k}_{p}" for p in range(self.oof[k].shape[1])])
                test = pd.DataFrame(data=self.ypred[k], columns=[f"{k}_{p}" for p in range(self.ypred[k].shape[1])])
            else:
                train_tmp = pd.DataFrame(data=self.oof[k], columns=[f"{k}_{p}" for p in range(self.oof[k].shape[1])])
                test_tmp = pd.DataFrame(data=self.ypred[k], columns=[f"{k}_{p}" for p in range(self.ypred[k].shape[1])]) 
                train = pd.concat([train, train_tmp], axis=1)
                test = pd.concat([test, test_tmp], axis=1)
        return train, test

    def calc_metric(self, y_true, y_pred):
        """
        calculate evaluation metric for each task
        this may need to be changed based on the metric of interest
        """

        if self.task == "multiclass":
            return f1_score(y_true, y_pred, average="macro")
        elif self.task == "binary":
            return roc_auc_score(y_true, y_pred)
        elif self.task == "regression":
            return np.sqrt(mean_squared_error(y_true, y_pred))

    def get_params(self, seed):
        """
        get parameters for linear model
        """

        if self.task == "regression": # Ridge
            params = {'alpha': 220, 'solver': 'lsqr', 'fit_intercept': True,
                'max_iter': 5000, 'random_state': seed}
        elif (self.task == "binary") | (self.task == "multiclass"): # logistic regression
            params = {'penalty': 'l2', 'tol' : 0.0001, 'C' : 1.0, 'fit_intercept' : True, 
                      'random_state' : seed, 'solver': 'lbfgs', 'max_iter' : 5000, 
                      'multi_class' : 'auto', 'verbose' : 0, 'warm_start' : False}
        return params

    def fit(self):
        """
        fit a linear model for stacking ensemble
        """

        # train, test
        train, test = self.data_formatter()
        y = self.yval

        # z-scoring
        scaler = StandardScaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)

        # train a linear model
        kf = KFold(n_splits=self.n_splits, random_state=self.seed, shuffle=True)
        stack_oof = np.zeros(len(y))
        stack_pred = np.zeros(test.shape[0])
        for fold, (train_index, test_index) in enumerate(kf.split(train)):
            # train test split
            x_train, x_val = train[train_index, :], train[test_index, :]
            y_train, y_val = y[train_index], y[test_index]

            # model fitting
            params = self.get_params(self.seed + fold)
            if self.task == "regression":
                mdl = linear_model.Ridge(**params)
                mdl.fit(x_train, y_train)
                stack_oof[test_index] = mdl.predict(x_val)
                stack_pred += mdl.predict(test) / self.n_splits
            elif (self.task == "binary") | (self.task == "multiclass"):
                mdl = linear_model.LogisticRegression(**params)
                mdl.fit(x_train, y_train)
                preds = mdl.predict_proba(x_val)
                if self.task == "binary":
                    stack_oof[test_index] = preds[:, 1]
                    stack_pred += mdl.predict_proba(test)[:, 1] / self.n_splits
                elif self.task == "multiclass":
                    stack_oof[test_index] = np.argmax(preds, axis=1)
                    stack_pred += np.argmax(mdl.predict_proba(test), axis=1) / self.n_splits
            
            # partial cv score
            score = self.calc_metric(y_val, stack_oof[test_index])
            print(f"fold {fold}: score = {score}")

        # overall cv score
        score = self.calc_metric(y, stack_oof)
        print(f"Overall CV score = {score}")

        return stack_oof, stack_pred
