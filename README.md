# DataScienceComp
EDA and modeling pipeline for Kaggle-like competitions

Note that this repository is under development so there may be some bugs.

# Basic usage
After cloning this repository, you define the following variables in your jupyter notebook:

- train_df: train pandas dataframe
- test_df: test pandas dataframe
- target: target column name (str)
- features: list of feature names
- categoricals: list of categorical feature names. Note that categoricals need to be in 'features'
- model: 'lgb', 'xgb', 'catb', 'linear', or 'nn'
- task: 'regression', 'multiclass', or 'binary'
- n_splits: K in KFold (default is 4)
- cv_method: 'KFold', 'StratifiedKFold', 'TimeSeriesSplit', 'GroupKFold', 'StratifiedGroupKFold'
- group: group feature name when GroupKFold or StratifiedGroupKFold are used (otherwise None)
- scaler: None, 'MinMax', 'Standard'

Then run the model like the following:

```python

# fit LGB regression model
model = RunModel(train_df, test_df, target, features,     
        categoricals=categoricals, target_encoding=False, model="lgb", 
        task="regression", n_splits=4, cv_method="KFold", 
        group=None, seed=116, scaler=None)


```

To visualize feature importance:

```python

# compute feature importance for visualization
sorted_feature_importance_df = model.plot_feature_importance()

```
