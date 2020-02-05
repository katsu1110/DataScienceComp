import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
from tqdm import tqdm

# Imputer
def nan2onehot(df, features):
    isnan_features = []
    for f in features:
        if df[f].isna().sum() > len(df) * 0.05:
            df[f + "_isnan"] = np.zeros(len(df))
            df.loc[(df[f].isna().values == True), f + "_isnan"] = 1
            isnan_features.append(f + "_isnan")
    return df, isnan_features

def target_encoding(x_train, x_test, label_train, cols, suffix = "_te", num_fold = 5, smooth_param = 0.001, stratified = False):
    # target encodingを行う関数

    # input:
    # smooth_param : smoothingの強さを決める． あるカテゴリのサンプルが少ないときに，それを全体のラベルの平均値をpriorとして均す
    
    x_train["label"] = label_train
    label_average = np.average(label_train) ## 全体の平均値． smoothingに使う
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
    if pre_concat: ##もしテストデータも用いてcountするなら
        dic = pd.concat([x_train[col], x_test[col]]).value_counts().to_dict()
    else:
        dic = x_train[col].value_counts().to_dict()
    x_train[col + suffix] = x_train[col].map(dic)
    x_test[col + suffix] = x_test[col].map(dic)
    return x_train, x_test



###########上の関数群を使ってencoding

def preprocess(x_train, x_test, y_train, continuous, categoricals, drops):
    
    ## target encoding
    x_train, x_test = target_encoding(x_train, x_test, y_train, categoricals, suffix = "_te", num_fold = 3, smooth_param = 0.01)
#         x_train, x_test = count_encodiing(x_train, x_test, col, suffix = "_count", pre_concat=True)
    x_train = x_train.drop(drops, axis = 1)
    x_test = x_test.drop(drops, axis = 1)
        
    return x_train, x_test
