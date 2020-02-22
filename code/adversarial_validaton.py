import os
import sys

import numpy as np
import pandas as pd

# custom modeling functions
mypath = os.getcwd()
sys.path.append(mypath + '/models/')
from base_models import BaseModel
from lgb_models import LgbModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, log_loss, roc_auc_score

from sklearn.model_selection import KFold, StratifiedKFold

# adversarial validation
class AdversarialValidation(object):
    """
    perform adversarial validation

    """
    def __init__(self, train, test, features, categoricals=[], group=None, n_splits=4):
        self.train = train
        self.test = test
        self.features = features
        self.categoricals = categoricals
        self.group = group
        self.n_splits = n_splits

    def calc_metric(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

    def run(self):
        # train = 1, test = 0
        df = pd.concat([self.train[self.features], self.test[self.features]],
                        ignore_index=True)
        df["is_train"] = 0
        df.loc[:self.train.shape[0], "is_train"] = 1

        # train, test split
        kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        tr_idx, va_idx = list(kf.split(df, df["is_train"]))[0]
        train = df.loc[tr_idx, :].reset_index(drop=True, inplace=False)
        test = df.loc[va_idx, :].reset_index(drop=True, inplace=False)

        # fit LGB
        if self.group is not None:
            cv_method = "GroupKFold"
        else:
            cv_method = "KFold"
        cls = LgbModel(train, test, "is_train", self.features, categoricals=self.categoricals, task="binary",
            group=self.group, n_splits=self.n_splits, cv_method=cv_method)

        return cls
