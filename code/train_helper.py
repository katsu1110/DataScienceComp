import numpy as np

def get_params(model="lgb", task="regression", seed=1220):
    """
    return hyperparameters of a given model and task
    """
    if model == "lgb":
        # list is here: https://lightgbm.readthedocs.io/en/latest/Parameters.html
        params = {
                    'n_estimators': 4000,
                    'objective': task,
                    'boosting_type': 'gbdt',
                    'num_leaves': 128,
                    'min_data_in_leaf': 64,
                    'max_depth': -1,
                    'learning_rate': 0.04,
                    'subsample': 0.76,
                    'subsample_freq': 1,
                    'feature_fraction': 0.2,
                    'seed': seed,
                    'early_stopping_rounds': 100
                    }    
        if task == "regression":
            params["metric"] = "rmse"
        elif task == "binary":
            params["metric"] = "auc" # binary_logloss
        elif task == "multiclass":
            params["metric"] = "multi_logloss" # cross_entropy, auc_mu

    elif model == "xgb":
        # list is here: https://xgboost.readthedocs.io/en/latest/parameter.html
        params = {
        'colsample_bytree': 0.2,                 
        'learning_rate': 0.08,
        'max_depth': 4,
        'subsample': 1,
        'min_child_weight': 4,
        'gamma':0.24,
        'seed': seed,
        'n_estimators':2000
        }
        if task == "regression":
            params["objective"] = 'reg:squarederror'
            params["eval_metric"] = "rmse"
        elif task == "binary":
            params["objective"] = 'binary:logistic'
            params["eval_metric"] = "auc"
        elif task == "multiclass":
            params["objective"] = 'multi:softmax'
            params["eval_metric"] = "mlogloss" 

    elif model == "catb":
        params = { 'task_type': "CPU",
                'learning_rate': 0.08, 
                'iterations': 2000,
                'colsample_bylevel': 0.2,
                'random_seed': seed,
                'use_best_model': True,
                'early_stopping_rounds': 100
                }
        if task == "regression":
            params["loss_function"] = "RMSE"
        elif (task == "binary") | (task == "multiclass"):
            params["loss_function"] = "Logloss"

    elif model == "linear":
        params = {
            'max_iter': 8000,
            'fit_intercept': True,
            'random_state': seed
        }

    elif model == "nn":
        # adapted from https://github.com/ghmagazine/kagglebook/blob/master/ch06/ch06-03-hopt_nn.py
        params = {
            'input_dropout': 0.0,
            'hidden_layers': 2,
            'hidden_units': 128,
            'embedding_out_dim': 4,
            # 'hidden_activation': 'relu', # use always mish
            'hidden_dropout': 0.08,
            # 'batch_norm': 'before_act', # use always LayerNormalization
            'optimizer': {'type': 'adam', 'lr': 1e-4},
            'batch_size': 128,
            'epochs': 80
        }
    return params
    
def get_oof_ypred(model, x_val, x_test, modelname="lgb", task="regression"):  
    """
    get oof and target predictions
    """

    if task == "binary": # classification
        # sklearn API
        if modelname in ["xgb", "catb", "linear"]:
            oof_pred = model.predict_proba(x_val)[1]
            y_pred = model.predict_proba(x_test)[1]
        else:
            oof_pred = model.predict(x_val)
            y_pred = model.predict(x_test)

            # NN specific
            if modelname == "nn":
                oof_pred = oof_pred.ravel()
                y_pred = y_pred.ravel()        

    elif task == "multitask":
        # sklearn API
        if modelname in ["xgb", "catb", "linear"]:
            oofs = model.predict_proba(x_val)
            ypred = model.predict_proba(x_test)
        else:
            oofs = model.predict(x_val)
            ypred = model.predict(x_test)

        oof_pred = np.argmax(oofs, axis=1)
        y_pred = np.argmax(ypred, axis=1)

    else: # regression
        oof_pred = model.predict(x_val)
        y_pred = model.predict(x_test)

        # NN specific
        if modelname == "nn":
            oof_pred = oof_pred.ravel()
            y_pred = y_pred.ravel()

    return oof_pred, y_pred