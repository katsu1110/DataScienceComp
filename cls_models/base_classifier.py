import numpy as np
import pandas as pd
import os, sys
mypath = os.getcwd()
sys.path.append(mypath + '/code/')
from cv_methods import GroupKFold, StratifiedGroupKFold
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from matplotlib import pyplot

from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

class ClassifierBase(object):
    """
    Base Classifier Class

    """

    def __init__(self, train_df, test_df, target, features, categoricals=[], n_splits=3, cv_method="KFold", group=None, verbose=True):
        self.train_df = train_df
        self.test_df = test_df
        self.target = target
        self.features = features
        self.n_splits = n_splits
        self.categoricals = categoricals
        self.cv_method = cv_method
        self.group = group
        self.cv = self.get_cv()
        self.verbose = verbose
        self.params = self.get_params()
        self.y_pred, self.score, self.model, self.oof, self.y_val, self.fi_df = self.fit()

    def train_model(self, train_set, val_set):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError

    def convert_x(self, x):
        return x

    def calc_metric(self, y_true, y_pred): # this may need to be changed based on the metric of interest
        return roc_auc_score(y_true, y_pred)

    def get_cv(self):
        if self.cv_method == "KFold":
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            return cv.split(self.train_df)
        elif self.cv_method == "StratifiedKFold":
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            return cv.split(self.train_df, self.train_df[self.target])
        elif self.cv_method == "TimeSeriesSplit":
            cv = TimeSeriesSplit(max_train_size=None, n_splits=self.n_splits)
            return cv.split(self.train_df)
        elif self.cv_method == "GroupKFold":
            cv = GroupKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            return cv.split(self.train_df, self.train_df[self.target], self.group)
        elif self.cv_method == "StratifiedGroupKFold":
            cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            return cv.split(self.train_df, self.train_df[self.target], self.group)

    def fit(self):
        # initialize
        oof_pred = np.zeros((self.train_df.shape[0], ))
        y_vals = np.zeros((self.train_df.shape[0], ))
        y_pred = np.zeros((self.test_df.shape[0], ))
        fi = np.zeros((self.n_splits, len(self.features)))

        for fold, (train_idx, val_idx) in enumerate(self.cv):
            # train test split
            x_train, x_val = self.train_df.loc[train_idx, self.features], self.train_df.loc[val_idx, self.features]
            y_train, y_val = self.train_df.loc[train_idx, self.target], self.train_df.loc[val_idx, self.target]

            # fitting & get feature importance
            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
            model, importance = self.train_model(train_set, val_set)
            fi[fold, :] = importance
            conv_x_val = self.convert_x(x_val)
            y_vals[val_idx] = y_val
            oof_pred[val_idx] = model.predict(conv_x_val).reshape(oof_pred[val_idx].shape)
            x_test = self.convert_x(self.test_df[self.features])
            y_pred += model.predict(x_test).reshape(y_pred.shape) / self.n_splits
            print('Partial score of fold {} is: {}'.format(fold, self.calc_metric(y_val, oof_pred[val_idx])))

        # feature importance data frame
        fi_df = pd.DataFrame()
        for n in np.arange(self.n_splits):
            tmp = pd.DataFrame()
            tmp["features"] = self.features
            tmp["importance"] = fi[n, :]
            tmp["fold"] = n
            fi_df = pd.concat([fi_df, tmp], ignore_index=True)
        gfi = fi_df[["features", "importance"]].groupby(["features"]).mean().reset_index()
        fi_df = fi_df.merge(gfi, on="features", how="left", suffixes=('', '_mean'))

        # outputs
        loss_score = self.calc_metric(self.train_df[self.target], oof_pred)
        if self.verbose:
            print('Our oof loss score is: ', loss_score)
        return y_pred, loss_score, model, oof_pred, y_vals, fi_df

    def plot_feature_importance(self, rank_range=[1, 50]):
        # plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 20))
        sns.barplot(data=self.fi_df.sort_values(by = "importance_mean", ascending=False).reset_index().iloc[self.n_splits * (rank_range[0]-1) : self.n_splits * rank_range[1]],
                    x ="importance", y ="features", orient='h')
        ax.set_xlabel("feature importance")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
