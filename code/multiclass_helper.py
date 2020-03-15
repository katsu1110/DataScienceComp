import numpy as np
import pandas as pd

def onehot_target(y, n_class):
    """
    make one-hot multiclass target 
    """
    y_cls = np.zeros((len(y), n_class))
    for i, v in enumerate(y):
        y_cls[i, v] = 1
    return y_cls

def ordinal_target(y, n_class):
    """
    make ordinal target
    """
    y_cls = np.zeros((len(y), n_class))
    for i, v in enumerate(y):
        y_cls[i, :v+1] = 1
    return y_cls