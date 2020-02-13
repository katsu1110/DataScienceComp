import os
import sys
import numpy as np
import pandas as pd
import random
from collections import Counter, defaultdict
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

class GroupKFold(object):
    """
    GroupKFold with random shuffle with a sklearn-like structure

    """

    def __init__(self, n_splits=4, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y, group):
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        unique_ids = X[group].unique()
        for fold, (tr_group_idx, va_group_idx) in enumerate(kf.split(unique_ids)):
            # split groups
            tr_groups, va_groups = unique_ids[tr_group_idx], unique_ids[va_group_idx]
            train_idx = np.where(X[group].isin(tr_groups))[0]
            val_idx = np.where(X[group].isin(va_groups))[0]
            yield train_idx, val_idx

class StratifiedGroupKFold(object):
    """
    StratifiedGroupKFold with random shuffle with a sklearn-like structure

    """

    def __init__(self, n_splits=4, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y, group):
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        unique_ids = X[group].unique()
        target_mean = np.zeros(len(unique_ids))
        for i, u in enumerate(unique_ids):
            target_mean[i] = y[X[group].values == u].mean()
        for fold, (tr_group_idx, va_group_idx) in enumerate(kf.split(unique_ids, target_mean)):
            # split groups
            tr_groups, va_groups = unique_ids[tr_group_idx], unique_ids[va_group_idx]
            train_idx = np.where(X[group].isin(tr_groups))[0]
            val_idx = np.where(X[group].isin(va_groups))[0]
            yield train_idx, val_idx
