import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error
from sklearn import linear_model
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import optuna

# keras
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, LambdaCallback
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.layers import Input, Layer, Dense, Concatenate, Reshape, Dropout, merge, Add, BatchNormalization, GaussianNoise
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras.layers import Layer
from keras.callbacks import *
import tensorflow as tf
import math

# # visualize
# import matplotlib.pyplot as plt
# import matplotlib.style as style
# import seaborn as sns
# from matplotlib import pyplot
# from matplotlib.ticker import ScalarFormatter
# sns.set_context("talk")
# style.use('fivethirtyeight')

# custom
mypath = os.getcwd()
sys.path.append(mypath + '/code/')
from train_helper import get_params, get_oof_ypred
from cv_methods import GroupKFold, StratifiedGroupKFold
from nn_utils import Mish, LayerNormalization, CyclicLR
from permutation_importance import PermulationImportance

class RunModel(object):
    """
    Model Fitting and Prediction Class:

    train_df : train pandas dataframe
    test_df : test pandas dataframe
    target : target column name (str)
    features : list of feature names
    categoricals : list of categorical feature names
    model : lgb, xgb, catb, linear, or nn
    task : options are ... regression, multiclass, or binary
    n_splits : K in KFold (default is 3)
    cv_method : options are ... KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold, StratifiedGroupKFold
    group : group feature name when GroupKFold or StratifiedGroupKFold are used
    parameter_tuning : bool, only for LGB
    seed : seed (int)
    scaler : options are ... None, MinMax, Standard
    verbose : bool
    """

    def __init__(self, train_df, test_df, target, features, categoricals=[],
                model="lgb", task="regression", n_splits=3, cv_method="KFold", 
                group=None, parameter_tuning=False, seed=1220, scaler=None, verbose=True):
        self.train_df = train_df
        self.test_df = test_df
        self.target = target
        self.features = features
        self.categoricals = categoricals
        self.model = model
        self.task = task
        self.n_splits = n_splits
        self.cv_method = cv_method
        self.group = group
        self.parameter_tuning = parameter_tuning
        self.seed = seed
        self.scaler = scaler
        self.verbose = verbose
        self.cv = self.get_cv()
        self.y_pred, self.score, self.model, self.oof, self.y_val, self.fi_df = self.fit()

    def train_model(self, train_set, val_set):
        # verbose
        verbosity = 2000 if self.verbose else 0

        # get hyperparameters
        params = get_params(model=self.model, task=self.task, seed=self.seed)

        # compile model
        if self.model == "lgb": # LGB             
            model = lgb.train(params, train_set, num_boost_round=2000, valid_sets=[train_set, val_set], verbose_eval=verbosity)
            fi = model.feature_importance(importance_type="gain")

        elif self.model == "xgb": # xgb
            if self.task == "regression":
                model = xgb.XGBRegressor(**params)
            elif (self.task == "binary") | (self.task == "multiclass"):
                model = xgb.XGBClassifier(**params)
            model.fit(train_set['X'], train_set['y'], eval_set=[(val_set['X'], val_set['y'])],
                           early_stopping_rounds=100, verbose=verbosity)
            fi = np.asarray(list(model.get_booster().get_score(importance_type='gain').values()))

        elif self.model == "catb": # catboost
            if self.task == "regression":
                model = CatBoostRegressor(**params)
            elif (self.task == "binary") | (self.task == "multiclass"):
                model = CatBoostClassifier(**params)
            model.fit(train_set['X'], train_set['y'], eval_set=(val_set['X'], val_set['y']),
                verbose=verbosity, cat_features=self.categoricals)
            fi = model.get_feature_importance()

        elif self.model == "linear": # linear model
            if self.task == "regression":
                # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
                model = linear_model.Ridge(**{'alpha': 220, 'solver': 'lsqr', 'fit_intercept': params['fit_intercept'],
                                        'max_iter': params['max_iter'], 'random_state': params['random_state']})
            elif (self.task == "binary") | (self.task == "multiclass"):
                # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
                model = linear_model.LogisticRegression(**{"C": 1.0, "fit_intercept": params['fit_intercept'], 
                                        "random_state": params['random_state'], "solver": "lbfgs", "max_iter": params['max_iter'], 
                                        "multi_class": 'auto', "verbose":0, "warm_start":False})
            model.fit(train_set['X'], train_set['y'])

            # permutation importance to get a feature importance (off in default)
            # fi = PermulationImportance(model, train_set['X'], train_set['y'], self.features)
            fi = np.zeros(len(self.features)) # no feature importance computed

        elif self.model == "nn": # neural network
            inputs = []
            n_neuron = params['hidden_units']

            # embedding for categorical features 
            if len(self.categoricals) > 0:
                embeddings = []
                embedding_out_dim = params['embedding_out_dim']
                for i in self.categoricals:
                    input_ = Input(shape=(1,))
                    embedding = Embedding(int(np.absolute(self.train_df[i]).max() + 1), embedding_out_dim, input_length=1)(input_)
                    embedding = Reshape(target_shape=(embedding_out_dim,))(embedding)
                    inputs.append(input_)
                    embeddings.append(embedding)
                input_numeric = Input(shape=(len(self.features) - len(self.categoricals),))
                embedding_numeric = Dense(n_neuron)(input_numeric)
                embedding_numeric = Mish()(embedding_numeric)
                inputs.append(input_numeric)
                embeddings.append(embedding_numeric)
                x = Concatenate()(embeddings)

            else: # no categorical features
                inputs = Input(shape=(len(self.features), ))
                x = Dense(n_neuron)(inputs)
                x = Mish()(x)
                x = Dropout(params['hidden_dropout'])(x)
                x = LayerNormalization()(x)
                
            # more layers
            for i in np.arange(params['hidden_layers'] - 1):
                x = Dense(n_neuron // (2 * (i+1)))(x)
                x = Mish()(x)
                x = Dropout(params['hidden_dropout'])(x)
                x = LayerNormalization()(x)
            
            # output
            if self.task == "regression":
                out = Dense(1, activation="linear", name = "out")(x)
                loss = "mse"
            elif self.task == "binary":
                out = Dense(1, activation='sigmoid', name = 'out')(x)
                loss = "binary_crossentropy"
            elif self.task == "multiclass":
                out = Dense(len(self.target), activation='softmax', name = 'out')(x)
                loss = "categorical_crossentropy"
            model = Model(inputs=inputs, outputs=out)

            # compile
            if params['optimizer']['type'] == 'adam':
                model.compile(loss=loss, optimizer=Adam(lr=params['optimizer']['lr'], beta_1=0.9, beta_2=0.999, decay=1e-04))
            elif params['optimizer']['type'] == 'sgd':
                model.compile(loss=loss, optimizer=SGD(lr=params['optimizer']['lr'], decay=1e-6, momentum=0.9))

            # callbacks
            er = EarlyStopping(patience=10, min_delta=1e-4, restore_best_weights=True, monitor='val_loss')
            ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
            history = model.fit(train_set['X'], train_set['y'], callbacks=[er, ReduceLR],
                                epochs=params['epochs'], batch_size=params['batch_size'],
                                validation_data=[val_set['X'], val_set['y']])

            # permutation importance to get a feature importance (off in default)
            # fi = PermulationImportance(model, train_set['X'], train_set['y'], self.features)
            fi = np.zeros(len(self.features)) # no feature importance computed
        
        return model, fi # fitted model and feature importance

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        if self.model == "lgb":
            train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)
            val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)
        else:
            train_set = {'X': x_train, 'y': y_train}
            val_set = {'X': x_val, 'y': y_val}
        return train_set, val_set

    def calc_metric(self, y_true, y_pred): # this may need to be changed based on the metric of interest
        if self.task == "multiclass":
            return log_loss(y_true, y_pred)
        elif self.task == "binary":
            return roc_auc_score(y_true, y_pred) # log_loss
        elif self.task == "regression":
            return np.sqrt(mean_squared_error(y_true, y_pred))

    def get_cv(self):
        # return cv.split
        if self.cv_method == "KFold":
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            return cv.split(self.train_df)
        elif self.cv_method == "StratifiedKFold":
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            return cv.split(self.train_df, self.train_df[self.target])
        elif self.cv_method == "TimeSeriesSplit":
            cv = TimeSeriesSplit(max_train_size=None, n_splits=self.n_splits)
            return cv.split(self.train_df)
        elif self.cv_method == "GroupKFold":
            cv = GroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            return cv.split(self.train_df, self.train_df[self.target], self.group)
        elif self.cv_method == "StratifiedGroupKFold":
            cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            return cv.split(self.train_df, self.train_df[self.target], self.group)

    def fit(self):
        # initialize
        if self.task == "multitask":
            n_class = len(self.target)
            oof_pred = np.zeros((self.train_df.shape[0], n_class))
            y_vals = np.zeros((self.train_df.shape[0], n_class))
            y_pred = np.zeros((self.test_df.shape[0], n_class))
        else:
            oof_pred = np.zeros((self.train_df.shape[0], ))
            y_vals = np.zeros((self.train_df.shape[0], ))
            y_pred = np.zeros((self.test_df.shape[0], ))

        # group does not kick in when group k fold is used
        if self.group is not None:
            if self.group in self.features:
                self.features.remove(self.group)
            if self.group in self.categoricals:
                self.categoricals.remove(self.group)
        fi = np.zeros((self.n_splits, len(self.features)))

        # scaling, if necessary
        if self.scaler is not None:
            # fill NaN
            numerical_features = [f for f in self.features if f not in self.categoricals]
            self.train_df[numerical_features] = self.train_df[numerical_features].fillna(self.train_df[numerical_features].median())
            self.test_df[numerical_features] = self.test_df[numerical_features].fillna(self.test_df[numerical_features].median())
            self.train_df[self.categoricals] = self.train_df[self.categoricals].fillna(self.train_df[self.categoricals].mode().iloc[0])
            self.test_df[self.categoricals] = self.test_df[self.categoricals].fillna(self.test_df[self.categoricals].mode().iloc[0])

            # scaling
            if self.scaler == "MinMax":
                scaler = MinMaxScaler()
            elif self.scaler == "Standard":
                scaler = StandardScaler()
            df = pd.concat([self.train_df[numerical_features], self.test_df[numerical_features]], ignore_index=True)
            scaler.fit(df[numerical_features])
            x_test = self.test_df.copy()
            x_test[numerical_features] = scaler.transform(x_test[numerical_features])
            if self.model == "nn":
                x_test = [np.absolute(x_test[i]) for i in self.categoricals] + [x_test[numerical_features]]
            else:
                x_test = x_test[self.features]
        else:
            x_test = self.test_df[self.features]

        # fitting with out of fold
        for fold, (train_idx, val_idx) in enumerate(self.cv):
            # train test split
            x_train, x_val = self.train_df.loc[train_idx, self.features], self.train_df.loc[val_idx, self.features]
            y_train, y_val = self.train_df.loc[train_idx, self.target], self.train_df.loc[val_idx, self.target]

            # fitting & get feature importance
            if self.scaler is not None:
                x_train[numerical_features] = scaler.transform(x_train[numerical_features])
                x_val[numerical_features] = scaler.transform(x_val[numerical_features])
                if self.model == "nn":
                    x_train = [np.absolute(x_train[i]) for i in self.categoricals] + [x_train[numerical_features]]
                    x_val = [np.absolute(x_val[i]) for i in self.categoricals] + [x_val[numerical_features]]

            # model fitting
            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
            model, importance = self.train_model(train_set, val_set)
            fi[fold, :] = importance
            y_vals[val_idx] = y_val

            # predictions
            oofs, ypred = get_oof_ypred(model, x_val, x_test, self.model, self.seed)
            oof_pred[val_idx] = oofs.reshape(oof_pred[val_idx].shape)
            y_pred += ypred.reshape(y_pred.shape) / self.n_splits

            # check cv score
            print('Partial score of fold {} is: {}'.format(fold, self.calc_metric(y_vals[val_idx], oof_pred[val_idx])))

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
        loss_score = self.calc_metric(y_vals, oof_pred)
        if self.verbose:
            print('Our oof loss score is: ', loss_score)
        return y_pred, loss_score, model, oof_pred, y_vals, fi_df

    def plot_feature_importance(self, rank_range=[1, 50]):
        # plot feature importance
        _, ax = plt.subplots(1, 1, figsize=(10, 20))
        sorted_df = self.fi_df.sort_values(by = "importance_mean", ascending=False).reset_index().iloc[self.n_splits * (rank_range[0]-1) : self.n_splits * rank_range[1]]
        sns.barplot(data=sorted_df, x ="importance", y ="features", orient='h')
        ax.set_xlabel("feature importance")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return sorted_df