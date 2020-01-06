class Base_Model(object):
    
    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=3, verbose=True):
        self.train_df = train_df
        self.test_df = test_df
        self.features = features
        self.n_splits = n_splits
        self.categoricals = categoricals
        self.target = 'accuracy_group'
        self.cv = self.get_cv()
        self.verbose = verbose
        self.params = self.get_params()
        self.y_pred, self.score, self.model, self.oof, self.y_val, self.fi_df = self.fit()
        
    def train_model(self, train_set, val_set):
        raise NotImplementedError
        
    def get_cv(self):
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        return cv.split(self.train_df, self.train_df[self.target])
        
    def get_params(self):
        raise NotImplementedError
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError
        
    def convert_x(self, x):
        return x
            
    def fit(self):
        oof_pred = np.zeros((self.train_df.shape[0], ))
        y_vals = np.zeros((self.train_df.shape[0], ))
        y_pred = np.zeros((self.test_df.shape[0], ))
        fi = np.zeros((self.n_splits, len(self.features)))
        
        # group KFold
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=71)
        unique_ids = self.train_df["installation_id"].unique()
        for fold, (tr_group_idx, va_group_idx) in enumerate(kf.split(unique_ids)):
            tr_groups, va_groups = unique_ids[tr_group_idx], unique_ids[va_group_idx]

            train_idx = self.train_df["installation_id"].isin(tr_groups)
            val_idx = self.train_df["installation_id"].isin(va_groups)
            
#         for fold, (train_idx, val_idx) in enumerate(self.cv):
#             x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]
#             y_train, y_val = self.train_df[self.target][train_idx], self.train_df[self.target][val_idx]
            x_train, x_val = self.train_df.loc[train_idx, self.features], self.train_df.loc[val_idx, self.features]
            y_train, y_val = self.train_df.loc[train_idx, self.target], self.train_df.loc[val_idx, self.target]
            
            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
            model, importance = self.train_model(train_set, val_set)
            fi[fold, :] = importance
            conv_x_val = self.convert_x(x_val)
            y_vals[val_idx] = y_val
            oof_pred[val_idx] = model.predict(conv_x_val).reshape(oof_pred[val_idx].shape)
            x_test = self.convert_x(self.test_df[self.features])
            y_pred += model.predict(x_test).reshape(y_pred.shape) / self.n_splits
            print('Partial score of fold {} is: {}'.format(fold, eval_qwk_lgb_regr(y_val, oof_pred[val_idx])[1]))
        
        # feature importance stored in data frame
        me = np.mean(fi, axis=0)
        ci = 1.96 * np.std(fi, axis=0) / np.sqrt(len(self.features))
        ranking = np.argsort(-me)
        fi_df = pd.DataFrame()
        fi_df["features"] = np.array(self.features)[ranking]
        fi_df["importance_mean"] = me[ranking]
        fi_df["importance_ci"] = ci[ranking]
        
        _, loss_score, _ = eval_qwk_lgb_regr(self.train_df[self.target], oof_pred)
        if self.verbose:
            print('Our oof cohen kappa score is: ', loss_score)
        return y_pred, loss_score, model, oof_pred, y_vals, fi_df
