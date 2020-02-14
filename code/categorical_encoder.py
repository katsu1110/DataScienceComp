import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import preprocessing

# label encoding object features
def label_encoding(x_train, x_test, cat_features):
    """
    label encoding object features
    """
    to_remove = []
    for c in cat_features:
        try:
            # label encoding
            le = preprocessing.LabelEncoder()
            x_train[c] = le.fit_transform(x_train[c].astype(str))
            x_test[c] = le.transform(x_test[c].astype(str))
            x_train[c] = x_train[c].astype(int)
            x_test[c] = x_test[c].astype(int)
        except: # cannot label encode = new value in test
            to_remove.append(c)
    cat_features = [c for c in cat_features if c not in to_remove]
    return x_train, x_test, cat_features

def target_encoding(x_train, x_test, label_train, cols, suffix = "_te", num_fold = 5, smooth_param = 0.001, stratified = False):
    # performs target encoding

    # input:
    # smooth_param : strength of smoothing. If samples of a certain category is in short, use target mean as a prior

    x_train["label"] = label_train
    label_average = np.average(label_train) ## overall mean for smoothing
    n_reg = len(x_train) * smooth_param

    for col in cols:
        if num_fold > 1:
            if stratified:
                kfold = StratifiedKFold(num_fold, shuffle=True, random_state=42)
            else:
                kfold = KFold(num_fold, shuffle=True, random_state=42)
            x_train[col + "_te"] = -999
            for k_fold, (agg_inds, not_agg_inds) in enumerate(kfold.split(x_train, x_train[col].astype("str"))):
                print(f"{col}, {k_fold+1} / {num_fold}")
                dic = x_train.iloc[agg_inds].groupby(col)["label"].agg(lambda x : (np.sum(x) + n_reg * label_average)/(len(x) + n_reg)).to_dict()
                x_train.loc[not_agg_inds, col + "_te"] = x_train.loc[not_agg_inds, col].map(dic)
            dic = x_train.groupby(col)["label"].agg("mean").to_dict()
        else:
            dic = x_train.groupby(col)["label"].agg("mean").to_dict()
            x_train[col + "_te"] = x_train[col].map(dic)
        x_test[col + "_te"] = x_test[col].map(dic)

    x_train = x_train.drop("label", axis = 1)
    return x_train, x_test

def count_encodiing(x_train, x_test, col, suffix = "_count", pre_concat = True):
    if pre_concat: ## if also with test data
        dic = pd.concat([x_train[col], x_test[col]]).value_counts().to_dict()
    else:
        dic = x_train[col].value_counts().to_dict()
    x_train[col + suffix] = x_train[col].map(dic)
    x_test[col + suffix] = x_test[col].map(dic)
    return x_train, x_test

########### encoding examples ###########

def preprocess(x_train, x_test, y_train, continuous, categoricals, drops):

    ## target encoding
    x_train, x_test = target_encoding(x_train, x_test, y_train, categoricals, suffix = "_te", num_fold = 3, smooth_param = 0.01)
#         x_train, x_test = count_encodiing(x_train, x_test, col, suffix = "_count", pre_concat=True)
    x_train = x_train.drop(drops, axis = 1)
    x_test = x_test.drop(drops, axis = 1)

    return x_train, x_test
