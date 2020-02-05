# basics
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from utils import row_col

# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib_venn import venn2
import seaborn as sns
from matplotlib import pyplot
import plotly.figure_factory as ff
import plotly.express as px
import missingno as msno

sns.set_context("talk")
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 13})
    
 # for categorical features
class CategoricalVisualizer():
    def __init__(self, train, test, features, target):
        self.train = train
        self.test = test
        self.features = features
        self.target = target

    def column_checker(self):
        '''

        EDA function to explore categorical variables in train & test data

        INPUT: train ... train (pandas dataframe)
           test ... test (pandas dataframe)
           features ... categorical features in train & test

        OUTPUT: chk ... pandas dataframe where columns are 
        "overlap", "train_nans", "test_nans", "train_nunique", "test_nunique"

        '''

        # data_type (numerical or categorical), n_nan, n_unique
        chk = pd.DataFrame()
        chk["features"] = self.features
        columns = ["overlap", "train_nans", "test_nans", "train_nunique", "test_nunique"]

        for c in columns:
            chk[c] = np.nan
        for i, f in enumerate(self.features):
            print("feature name = " + f + ", dtype = " + self.train[f].dtype)

        # overlap between train & test
        chk.loc[i, "overlap"] = len(set(self.train[f].values.tolist() + self.test[f].values.tolist())) / self.train[f].nunique()

        # nans
        chk.loc[i, "train_nans"] = self.train[f].isna().sum()
        chk.loc[i, "test_nans"] = self.test[f].isna().sum()

        # nuniques
        chk.loc[i, "train_nunique"] = self.train[f].nunique()
        chk.loc[i, "test_nunique"] = self.test[f].nunique()

        for c in ["train_nans", "test_nans", "train_nunique", "test_nunique"]:
            chk[c] = chk[c].astype(int)
        return chk

    # barplot for categoricals
    def plot_bars(self):
        """
        plot vs target for each categorical feature with a count plot  
        """

        # plot
        nrow, ncol = row_col(len(self.features))
        fig, ax = plt.subplots(nrow, ncol, figsize=(7 * ncol, 5 * nrow))
        ax = ax.flatten()
        for i, f in enumerate(self.features):
            # count plot
            trainc = pd.DataFrame(self.train[f].value_counts()).reset_index()
            testc = pd.DataFrame(self.test[f].value_counts()).reset_index()
            cats = list(set(trainc["index"].values.tolist() + testc["index"].values.tolist()))
            for c in cats:
                if c in trainc["index"].values.tolist():
                    ax[i].bar(np.arange(len(cats)) - 0.2, trainc.loc[trainc["index"] == c, f], color="k", width=0.2, alpha=0.5)
                if c in testc["index"].values.tolist():
                    ax[i].bar(np.arange(len(cats)), testc.loc[testc["index"] == c, f], color="r", width=0.2, alpha=0.5)
            ax[i].set_title(f)
            ax[i].set_xticks(np.arange(len(cats)))
            ax[i].set_xticklabels(cats, rotation=45, ha="right")
            
            # target mean
            ax2 = ax[i].twinx()
            for k, c in enumerate(cats):
                if c in trainc["index"].values.tolist():
                    me = np.nanmean(self.train.loc[self.train[f] == c, self.target].values)
                    sem = np.nanstd(self.train.loc[self.train[f] == c, self.target].values) / np.sum(self.train[f] == c)
                    ax2.errorbar(k, me, sem, marker='o', color="g", alpha=0.5)
                    
        for a in ax[len(self.features):]:
            a.axis("off")
        plt.tight_layout()

    # venn plot
    def plot_venn(self):
        """
        see overlap of categorical variables btw. train & test
        """

        # plot
        nrow, ncol = row_col(self)
        fig, ax = plt.subplots(nrow, ncol, figsize=(7 * ncol, 5 * nrow))
        ax = ax.flatten()
        for i, f in enumerate(self.features):
            a = train[f].values.tolist()
            b = test[f].values.tolist()
            venn2([set(a), set(b)], set_labels = ('train', 'test'), ax=ax[i])
            plt.title(feature)
        plt.tight_layout()