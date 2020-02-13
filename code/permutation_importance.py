import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

class PermulationImportance(object):
    """
    compute permutation importance

    """
    def __init__(self, model, X, y, features):
        self.model = model
        self.X = X
        self.y = y
        self.features = features

    def calc_metric(self):
        "smaller value is better fit"
        return mean_absolute_error(self.model.predict(self.X)[0], self.y)

    def run(self):
        baseline = self.calc_metric(self)
        imp = []
        for f in self.features:
            idx = self.features.index(f)
            save = self.X[:, idx].copy()
            self.X[:, idx] = np.random.permutation(self.X[:, idx])
            m = self.calc_metric(self)
            self.X[:, idx] = save
            imp.append(baseline - m)
        return np.array(imp)
