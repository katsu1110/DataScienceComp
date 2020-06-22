import numpy as np
import pandas as pd
import os, sys
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, f1_score
import lightgbm as lgb

# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
sns.set_context("talk")
style.use('seaborn-colorblind')

# adversarial validation
class AdversarialValidation(object):
    """
    perform adversarial validation using LGB model

    :INPUTS:
    
    :train: train pandas dataframe
    :test: test pandas dataframe
    :features: list of feature names
    :n_splits: K in KFold (default is 4)
    
    :EXAMPLE:
    
    # run adversarial validation
    adv = AdversarialValidation(train, test, features, n_splits=4)
    # plot features different between train and test
    sorted_df = adv.plot_feature_importance()
    sorted_df.head()
    
    """
    
    def __init__(self, train : pd.DataFrame, test : pd.DataFrame, features : List, n_splits : int=4, seed : int=116):
        self.train = train
        self.test = test
        self.features = features
        self.n_splits = n_splits
        self.seed = seed
        self.fi_df = self.run()

    def calc_metric(self, y_true, y_pred):
        """
        compute AUC
        """
        return roc_auc_score(y_true, y_pred)

    def get_params(self):
        """
        LGB parameters (fast ones)
        """
        # list is here: https://lightgbm.readthedocs.io/en/latest/Parameters.html
        params = {
                    'n_estimators': 2000,
                    'objective': 'binary',
                    'boosting_type': 'gbdt',
                    'num_leaves': 16,
                    'learning_rate': 0.12,
                    'feature_fraction': 0.64,
                    'lambda_l1': 0.8,
                    'lambda_l2': 0.8,
                    'seed': self.seed,
                    'early_stopping_rounds': 80,
                    'num_boost_round': 2000
                    }    
        params["metric"] = "auc" # other candidates: binary_logloss
        # params["is_unbalance"] = True # assume unbalanced data

        return params
            
    def run(self):
        """
        run adversarial validation
        """

        # train = 1, test = 0
        target = "is_train"
        df = pd.concat([self.train[self.features], self.test[self.features]],
                        ignore_index=True)
        df[target] = 0
        df.loc[:self.train.shape[0], target] = 1

        # fit LGB
        params = self.get_params()
        kf = StratifiedKFold(n_splits=self.n_splits, random_state=self.seed, shuffle=True)
        fi = np.zeros((self.n_splits, len(self.features)))
        oof_pred = np.zeros(df.shape[0])
        for fold, (train_index, test_index) in enumerate(kf.split(df, df[target])):
            # train test split
            x_train, x_val = df.loc[train_index, self.features], df.loc[test_index, self.features]
            y_train, y_val = df.loc[train_index, target], df.loc[test_index, target]

            # model fitting
            train_set = lgb.Dataset(x_train, y_train)
            val_set = lgb.Dataset(x_val, y_val)
            model = lgb.train(params, train_set, valid_sets=[train_set, val_set], verbose_eval=100)
            fi[fold, :] = model.feature_importance(importance_type="gain")

            # predict
            oof_pred[test_index] = model.predict(x_val)
            print('Partial score of fold {} is: {}'.format(fold, self.calc_metric(y_val, oof_pred[test_index])))

        score = self.calc_metric(df[target], oof_pred)
        print('Our oof cv score (AUC) is: ', score)

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

        return fi_df

    def plot_feature_importance(self, rank_range=[1, 50]):
        """
        function for plotting feature importance
        """

        # plot feature importance
        _, ax = plt.subplots(1, 1, figsize=(10, 20))
        sorted_df = self.fi_df.sort_values(by = "importance_mean", ascending=False).reset_index().iloc[self.n_splits * (rank_range[0]-1) : self.n_splits * rank_range[1]]
        sns.barplot(data=sorted_df, x ="importance", y ="features", orient='h')
        ax.set_xlabel("feature importance")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return sorted_df