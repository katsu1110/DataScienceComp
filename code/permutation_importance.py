import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, log_loss, roc_auc_score
from scipy.stats import spearmanr

class PermulationImportance(object):
    """
    compute permutation importance

    """
    
    def __init__(self, model, X, y, features, task="regression"):
        self.model = model
        self.X = X
        self.y = y
        self.features = features
        self.task = task
    
    def calc_metric(self): # this may need to be changed based on the metric of interest
        if (self.task == "multiclass") | (self.task == "binary"):
            return log_loss(self.y, self.model.predict(self.X))
        elif self.task == "regression":
            return np.sqrt(mean_squared_error(self.y, self.model.predict(self.X)))

    def run(self):
        baseline = self.calc_metric()
        imp = []
        for f in self.features:
            idx = self.features.index(f)
            save = self.X[:, idx].copy()
            self.X[:, idx] = np.random.permutation(self.X[:, idx])
            m = self.calc_metric()
            self.X[:, idx] = save
            imp.append(baseline - m)
        return -np.array(imp)
