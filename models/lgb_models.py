import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.visualization import plot_optimization_history
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error
from base_models import BaseModel

class LgbModel(BaseModel):
    """
    LGB wrapper

    """

    def train_model(self, train_set, val_set):
        verbosity = 1000 if self.verbose else 0
        model = lgb.train(self.params, train_set, num_boost_round = 3000, valid_sets=[train_set, val_set], verbose_eval=verbosity)
        fi = model.feature_importance(importance_type="gain")
        return model, fi

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)
        val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)
        return train_set, val_set

    def convert_x(self, x):
        return x

    def get_params(self):
        # fast fit parameters
        params = {
                    'n_estimators': 4000,
                    'objective': self.task,
                    'boosting_type': 'gbdt',
                    'min_data_in_leaf': 50,
                    'max_depth': -1,
                    'learning_rate': 0.01,
                    'subsample': 0.75,
                    'subsample_freq': 1,
                    'feature_fraction': 0.9,
                    'lambda_l1': 1,
                    'lambda_l2': 1,
                    'seed': self.seed,
                    'early_stopping_rounds': 100
                    }

        # list is here: https://lightgbm.readthedocs.io/en/latest/Parameters.html
        if self.task == "regression":
            params["metric"] = "rmse"
        elif self.task == "binary":
            params["metric"] = "auc" # binary_logloss
        elif self.task == "multiclass":
            params["metric"] = "multi_logloss" # cross_entropy, auc_mu
        
        # Bayesian Optimization by Optuna
        if self.parameter_tuning == True:
            # define objective function
            def objective(trial):
                # train, test split
                train_x, test_x, train_y, test_y = train_test_split(self.train_df[self.features], 
                                                                    self.train_df[self.target],
                                                                    test_size=0.3, random_state=self.seed)
                dtrain = lgb.Dataset(train_x, train_y, categorical_feature=self.categoricals)
                dtest = lgb.Dataset(test_x, test_y, categorical_feature=self.categoricals)

                # parameters to be explored
                hyperparams = {'num_leaves': trial.suggest_int('num_leaves', 24, 1024),
                        'boosting_type': 'gbdt',
                        'objective': params["objective"],
                        'metric': params["metric"],
                        'max_depth': trial.suggest_int('max_depth', 4, 16),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                        'early_stopping_rounds': 100
                        }

                # LGB
                model = lgb.train(hyperparams, dtrain, valid_sets=dtest, verbose_eval=500)
                pred = model.predict(test_x)
                if (self.task == "binary") | (self.task == "multiclass"):
                    return log_loss(test_y, pred)
                elif self.task == "regression":
                    return np.sqrt(mean_squared_error(test_y, pred))

            # run optimization
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=50)

            print('Number of finished trials: {}'.format(len(study.trials)))
            print('Best trial:')
            trial = study.best_trial
            print('  Value: {}'.format(trial.value))
            print('  Params: ')
            for key, value in trial.params.items():
                print('    {}: {}'.format(key, value))

            params = trial.params

            # lower learning rate for better accuracy
            params["learning_rate"] = 0.001

            # plot history
            plot_optimization_history(study)

        return params