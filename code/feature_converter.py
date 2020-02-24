import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

# categorize features into dense or categorical
def categorize_features(df, target, cat_threshold=12):
    cat_features = []
    dense_features = []
    for f in df.columns.values.tolist():
        if f != target:
            if (df[f].dtype == "object") | (df[f].dtype == "bool") | (df[f].nunique() <= cat_threshold):
                cat_features.append(f)
            else:
                dense_features.append(f)
    features = dense_features + cat_features
    print(f"There are {len(features)} features.")
    print(f"There are {len(dense_features)} dense features.")
    print(f"There are {len(cat_features)} categorical features.")
    return features, dense_features, cat_features

# one-hot encoding nan
def nan2onehot(df, features):
    isnan_features = []
    for f in features:
        if df[f].isna().sum() > len(df) * 0.05:
            df[f + "_isnan"] = np.zeros(len(df))
            df.loc[(df[f].isna().values == True), f + "_isnan"] = 1
            isnan_features.append(f + "_isnan")
    return df, isnan_features

# outlier remover
def clipper(df, features):
    p01 = df[features].quantile(0.01)
    p99 = df[features].quantile(0.99)
    df[features] = df[features].clip(p01, p99, axis=1)
    return df

# to normal dist
def to_normal(train, test, features, method="yeo-johnson"):
    # method can be box-cox
    pt = PowerTransformer(method=method)
    train[features] = pt.fit_transform(train[features])
    test[features] = pt.transform(test[features])
    return train, test

# remove correlated features
def remove_correlated_features(df, features, threshold=0.999):
    counter = 0
    to_remove = []
    for feat_a in features:
        for feat_b in features:
            if feat_a != feat_b and feat_a not in to_remove and feat_b not in to_remove:
                c = np.corrcoef(df[feat_a], df[feat_b])[0][1]
                if c > threshold:
                    counter += 1
                    to_remove.append(feat_b)
                    print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(counter, feat_a, feat_b, c))
    return to_remove
