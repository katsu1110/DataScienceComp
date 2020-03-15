import numpy as np
  
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

    elif task == "multiclass":
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