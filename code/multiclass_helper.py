import numpy as np
import pandas as pd

def onehot_target(y):
    """
    make one-hot multiclass target 
    """
    y_cls = np.zeros((len(y), 4))
    for i, v in enumerate(y):
        y_cls[i, v] = 1
    return y_cls

def ordinal_target(y):
    """
    make ordinal target
    """
    y_cls = np.zeros((len(y), 4))
    for i, v in enumerate(y):
        y_cls[i, :v+1] = 1
    return y_cls