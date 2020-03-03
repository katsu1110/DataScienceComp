import os
import sys
import numpy as np
import pandas as pd
import random
from collections import Counter, defaultdict
from sklearn import model_selection

class GroupKFold(object):
    """
    GroupKFold with random shuffle with a sklearn-like structure

    """

    def __init__(self, n_splits=4, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, group=None):
        return self.n_splits

    def split(self, X, y, group):
        kf = model_selection.KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        unique_ids = X[group].unique()
        for fold, (tr_group_idx, va_group_idx) in enumerate(kf.split(unique_ids)):
            # split group
            tr_group, va_group = unique_ids[tr_group_idx], unique_ids[va_group_idx]
            train_idx = np.where(X[group].isin(tr_group))[0]
            val_idx = np.where(X[group].isin(va_group))[0]
            yield train_idx, val_idx

class StratifiedGroupKFold(object):
    """
    StratifiedGroupKFold with random shuffle with a sklearn-like structure

    """

    def __init__(self, n_splits=4, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, group=None):
        return self.n_splits

    def split(self, X, y, group):
        # Preparation
        max_y = np.max(y)
        groups = X[group].values

        # y counts per group
        unique_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(unique_num))
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
        kf = model_selection.GroupKFold(n_splits=self.n_splits)

        for train_idx, val_idx in kf.split(X, y, groups):
            # Training dataset and validation dataset
            id_train = X.loc[train_idx, group].unique()
            x_val, y_val = X.loc[val_idx, :], y.iloc[val_idx]
            id_val = x_val[group].unique()

            # y counts of training dataset and validation dataset
            y_counts_train = np.zeros(max_y+1)
            y_counts_val = np.zeros(max_y+1)
            for id_ in id_train:
                y_counts_train += y_counts_per_group[id_]
            for id_ in id_val:
                y_counts_val += y_counts_per_group[id_]

            # Determination ratio of validation dataset
            numratio_train = y_counts_train / np.max(y_counts_train)
            stratified_count = np.ceil(y_counts_val[np.argmax(y_counts_train)] * numratio_train)
            stratified_count = stratified_count.astype(int)

            # Select validation dataset randomly
            val_idx = np.array([])
            np.random.seed(self.random_state) 
            for num in range(max_y+1):
                val_idx = np.append(val_idx, np.random.choice(y_val[y_val==num].index, stratified_count[num]))
            val_idx = val_idx.astype(int)
            
            yield train_idx, val_idx
