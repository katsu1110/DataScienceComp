# basics
import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, f1_score, r2_score

# model
import xgboost as xgb
import operator

# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
import waterfall_chart
sns.set_context("talk")
style.use('fivethirtyeight')

class FeatureContribution(object):
    """
    Model Fitting and Prediction Class:

    df : pandas dataframe
    target : target column name (str)
    features : list of feature names
    model : currently only xgb
    task : options are ... regression, multiclass, or binary
    seed : seed (int)
    """

    def __init__(self, df, target, features, task="regression", seed=1220):
        self.df = df
        self.target = target
        self.features = features # needs to be in a desired order
        self.task = task
        self.seed = seed
        self.metric_score, self.model, self.fi = self.fit()

    def get_train_test(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.df[self.features], self.df[self.target], test_size=0.3, random_state=self.seed)
        return X_train, X_test, y_train, y_test

    def fit_model(self, X_train, X_test, y_train, y_test, features):
        # list is here: https://xgboost.readthedocs.io/en/latest/parameter.html
        params = {
            'colsample_bytree': 1,                 
            'learning_rate': 0.12,
            'max_depth': 4,
            'subsample': 1,
            'min_child_weight': 4,
            'gamma':0.24,
            'seed': self.seed,
            'n_estimators':400
        }
        if self.task == "regression":
            params["objective"] = 'reg:squarederror'
            params["eval_metric"] = "rmse"
        elif self.task == "binary":
            params["objective"] = 'binary:logistic'
            params["eval_metric"] = "auc"
        elif self.task == "multiclass":
            params["objective"] = 'multi:softmax'
            params["eval_metric"] = "mlogloss" 

        # modeling
        if self.task == "regression":
            model = xgb.XGBRegressor(**params)
        elif (self.task == "binary") | (self.task == "multiclass"):
            model = xgb.XGBClassifier(**params)
        model.fit(X_train[features], y_train, eval_set=[(X_test[features], y_test)],
                        early_stopping_rounds=80, verbose=200)

        # feature importance
        importance = model.get_booster().get_score(importance_type='gain')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()
        fi = np.zeros(len(features))
        for i, f in enumerate(features):
            try:
                fi[i] = df.loc[df['feature'] == f, "fscore"].iloc[0]
            except: # ignored by XGB
                continue

        return model, fi


    def calc_metric(self, y_true, y_pred): # this may need to be changed based on the metric of interest
        if self.task == "multiclass":
            return roc_auc_score(y_true, y_pred, average="macro")
        elif self.task == "binary":
            return roc_auc_score(y_true, y_pred) 
        elif self.task == "regression":
            return r2_score(y_true, y_pred)

    def fit(self):
        metric_score = np.zeros(len(self.features))
        X_train, X_test, y_train, y_test = self.get_train_test()
        features_to_use = []
        for i, f in enumerate(self.features):
            features_to_use.append(f)
            print("used features...", features_to_use)
            model, fi = self.fit_model(X_train, X_test, y_train, y_test, features_to_use)
            if self.task == "binary":
                y_pred = model.predict_proba(X_test[features_to_use])
            else:
                y_pred = model.predict(X_test[features_to_use])
            metric_score[i] = self.calc_metric(y_test, y_pred)
        return metric_score, model, fi
        
    def plot_feature_contribution(self):
        # plot feature contribution
        wc = waterfall_chart.plot(self.features, self.metric_score)
        return wc
