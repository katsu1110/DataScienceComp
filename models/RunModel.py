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
        self.params = self.get_params()
        self.y_pred, self.score, self.model, self.oof, self.y_val, self.fi_df = self.fit()

    def train_model(self, train_set, val_set):
        # verbose
        verbosity = 1000 if self.verbose else 0

        # compile model
        if self.model == "lgb": # LGB             
            model = lgb.train(self.params, train_set, num_boost_round=3000, valid_sets=[train_set, val_set], verbose_eval=verbosity)
            fi = model.feature_importance(importance_type="gain")

        elif self.model == "xgb": # xgb
            model = xgb.train(self.params, train_set, 
                         num_boost_round=2000, evals=[(train_set, 'train'), (val_set, 'val')], 
                         verbose_eval=verbosity, early_stopping_rounds=100)
            fi = np.asarray(list(model.get_score(importance_type='gain').values()))

        elif self.model == "catb": # catboost
            if self.task == "regression":
                model = CatBoostRegressor(**self.params)
            elif (self.task == "binary") | (self.task == "multiclass"):
                model = CatBoostClassifier(**self.params)
            model.fit(train_set['X'], train_set['y'], eval_set=(val_set['X'], val_set['y']),
                verbose=verbosity, cat_features=self.categoricals)
            fi = model.get_feature_importance()

        elif self.model == "linear": # linear model
            if self.task == "regression":
                # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
                model = linear_model.Ridge(**{'alpha': 220, 'solver': 'lsqr', 'fit_intercept': self.params['fit_intercept'],
                                        'max_iter': self.params['max_iter'], 'random_state': self.params['random_state']})
            elif (self.task == "binary") | (self.task == "multiclass"):
                # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
                model = linear_model.LogisticRegression(C=1.0, fit_intercept=self.params['fit_intercept'], class_weight='balanced',
                                        random_state=self.params['random_state'], solver='lbfgs', max_iter=self.params['max_iter'], 
                                        multi_class='auto', verbose=0, warm_start=False)
            model.fit(train_set['X'], train_set['y'])

            # permutation importance to get a feature importance (off in default)
            # fi = PermulationImportance(model, train_set['X'], train_set['y'], self.features)
            fi = np.zeros(len(self.features)) # no feature importance computed

        elif self.model == "nn": # neural network
            inputs = []
            n_neuron = self.params['hidden_units']
            if len(self.categoricals) > 0:
                embeddings = []
                embedding_out_dim = self.params['embedding_out_dim']
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
                x = Dropout(self.params['hidden_dropout'])(x)
                x = LayerNormalization()(x)
                
            # more layers
            for i in np.arange(self.params['hidden_layers'] - 1):
                x = Dense(n_neuron // (2 * (i+1)))(x)
                x = Mish()(x)
                x = Dropout(self.params['hidden_dropout'])(x)
                x = LayerNormalization()(x)
            
            # output
            if self.task == "regression":
                out = Dense(1, activation="linear", name = "out")(x)
                loss = "mse"
            elif self.task == "binary":
                out = Dense(1, activation='sigmoid', name = 'out')(x)
                loss = "binary_crossentropy"
            elif self.task == "multiclass":
                out = Dense(train_set['y'].nunique(), activation='softmax', name = 'out')(x)
                loss = "categorical_crossentropy"
            model = Model(inputs=inputs, outputs=out)

            # compile
            if self.params['optimizer']['type'] == 'adam':
                model.compile(loss=loss, optimizer=Adam(lr=self.params['optimizer']['lr'], beta_1=0.9, beta_2=0.999, decay=1e-04))
            elif self.params['optimizer']['type'] == 'sgd':
                model.compile(loss=loss, optimizer=SGD(lr=self.params['optimizer']['lr'], decay=1e-6, momentum=0.9))

            # callbacks
            er = EarlyStopping(patience=10, min_delta=1e-4, restore_best_weights=True, monitor='val_loss')
            ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
            history = model.fit(train_set['X'], train_set['y'], callbacks=[er, ReduceLR],
                                epochs=self.params['epochs'], batch_size=self.params['batch_size'],
                                validation_data=[val_set['X'], val_set['y']])

            # permutation importance to get a feature importance (off in default)
            # fi = PermulationImportance(model, train_set['X'], train_set['y'], self.features)
            fi = np.zeros(len(self.features)) # no feature importance computed
        
        return model, fi # fitted model and feature importance

    def get_params(self):
        if self.model == "lgb":
            # list is here: https://lightgbm.readthedocs.io/en/latest/Parameters.html
            params = {
                        'n_estimators': 4000,
                        'objective': self.task,
                        'boosting_type': 'gbdt',
                        'min_data_in_leaf': 50,
                        'max_depth': -1,
                        'learning_rate': 0.03,
                        'subsample': 0.75,
                        'subsample_freq': 1,
                        'feature_fraction': 0.9,
                        'seed': self.seed,
                        'early_stopping_rounds': 100
                        }            
            if self.task == "regression":
                params["metric"] = "rmse"
            elif self.task == "binary":
                params["metric"] = "auc" # binary_logloss
            elif self.task == "multiclass":
                params["metric"] = "multi_logloss" # cross_entropy, auc_mu
        elif self.model == "xgb":
            # list is here: https://xgboost.readthedocs.io/en/latest/parameter.html
            params = {
            'colsample_bytree': 0.8,                 
            'learning_rate': 0.05,
            'max_depth': 10,
            'subsample': 1,
            'min_child_weight':3,
            'gamma':0.25,
            'seed': self.seed,
            'n_estimators':4000
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
        elif self.model == "catb":
            params = { 'task_type': "CPU",
                   'learning_rate': 0.03, 
                   'iterations': 3000,
                   'random_seed': self.seed,
                   'use_best_model': True,
                   'early_stopping_rounds': 100
                    }
            if self.task == "regression":
                params["loss_function"] = "RMSE"
            elif (self.task == "binary") | (self.task == "multiclass"):
                params["loss_function"] = "Logloss"
        elif self.model == "linear":
            params = {
                'max_iter': 5000,
                'fit_intercept': True,
                'random_state': self.seed
            }
        elif self.model == "nn":
            # adapted from https://github.com/ghmagazine/kagglebook/blob/master/ch06/ch06-03-hopt_nn.py
            params = {
                'input_dropout': 0.0,
                'hidden_layers': 2,
                'hidden_units': 128,
                'embedding_out_dim': 4,
                # 'hidden_activation': 'relu', # use always mish
                'hidden_dropout': 0.05,
                # 'batch_norm': 'before_act', # use always LayerNormalization
                'optimizer': {'type': 'adam', 'lr': 1e-4},
                'batch_size': 256,
                'epochs': 80
            }
        return params

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        if self.model == "lgb":
            train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)
            val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)
        elif self.model == "xgb":
            train_set = xgb.DMatrix(x_train, y_train)
            val_set = xgb.DMatrix(x_val, y_val)
        else:
            train_set = {'X': x_train, 'y': y_train}
            val_set = {'X': x_val, 'y': y_val}
        return train_set, val_set

    def convert_x(self, x):
        if self.model == "xgb":
            return xgb.DMatrix(x)
        else:
            return x

    def calc_metric(self, y_true, y_pred): # this may need to be changed based on the metric of interest
        if self.task == "multiclass":
            return log_loss(y_true, y_pred)
        elif self.task == "binary":
            return roc_auc_score(y_true, y_pred)
        elif self.task == "regression":
            return np.sqrt(mean_squared_error(y_true, y_pred))

    def get_cv(self):
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
        oof_pred = np.zeros((self.train_df.shape[0], ))
        y_vals = np.zeros((self.train_df.shape[0], ))
        y_pred = np.zeros((self.test_df.shape[0], ))
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
            x_test = [np.absolute(x_test[i]) for i in self.categoricals] + [x_test[numerical_features]]
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
                x_train = [np.absolute(x_train[i]) for i in self.categoricals] + [x_train[numerical_features]]
                x_val = [np.absolute(x_val[i]) for i in self.categoricals] + [x_val[numerical_features]]
            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
            model, importance = self.train_model(train_set, val_set)
            fi[fold, :] = importance
            conv_x_val = self.convert_x(x_val)
            y_vals[val_idx] = y_val
            x_test = self.convert_x(x_test)
            if (self.model == "linear") & (self.task != "regression"):
                oofs = model.predict_proba(conv_x_val)
                ypred = model.predict_proba(x_test) / self.n_splits
            else:
                oofs = model.predict(conv_x_val)
                ypred = model.predict(x_test) / self.n_splits
            try:
                if oofs.shape[1] == 2:
                    oof_pred[val_idx] = oofs[:, -1]
                    y_pred += ypred[:, -1]
                elif oofs.shape[1] > 2:
                    oof_pred[val_idx] = np.argmax(oofs, axis=1)
                    y_pred += np.argmax(ypred, axis=1)
            except:
                oof_pred[val_idx] = oofs.reshape(oof_pred[val_idx].shape)
                y_pred += ypred
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
        loss_score = self.calc_metric(y_vals, oof_pred)
        if self.verbose:
            print('Our oof loss score is: ', loss_score)
        return y_pred, loss_score, model, oof_pred, y_vals, fi_df

    def plot_feature_importance(self, rank_range=[1, 50]):
        # plot
        _, ax = plt.subplots(1, 1, figsize=(10, 20))
        sorted_df = self.fi_df.sort_values(by = "importance_mean", ascending=False).reset_index().iloc[self.n_splits * (rank_range[0]-1) : self.n_splits * rank_range[1]]
        sns.barplot(data=sorted_df, x ="importance", y ="features", orient='h')
        ax.set_xlabel("feature importance")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return sorted_df
