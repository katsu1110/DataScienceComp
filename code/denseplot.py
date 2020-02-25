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
from matplotlib.ticker import ScalarFormatter
import plotly.figure_factory as ff
import plotly.express as px
import missingno as msno

sns.set_context("talk")
# sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
style.use('fivethirtyeight')
# plt.rcParams.update({'font.size': 13})
    
# for dense featuers
class DenseVisualizer():
    def __init__(self, train, test, features, target):
        self.train = train
        self.test = test
        self.features = features
        self.target = target

     # dense column checker
    def column_checker(self):
        '''

        EDA function to explore categorical variables in train & test data

        INPUT: train ... train (pandas dataframe)
                   test ... test (pandas dataframe)
                   features ... numerical features in train & test

        OUTPUT: chk ... pandas dataframe where columns are 
                          "train_nans", "test_nans", "train_nunique", "test_nunique", 
                          "train_min", "test_min", "train_max", "test_max", "train_mean", "test_mean", 
                          "train_skew", "test_skew"

        '''

        chk = pd.DataFrame()
        chk["features"] = self.features
        columns = ["train_nans", "test_nans", "train_nunique", "test_nunique", 
                          "train_min", "test_min", "train_max", "test_max", "train_mean", "test_mean", 
                          "train_skew", "test_skew"]
        for c in columns:
            chk[c] = np.nan
        for i, f in enumerate(self.features):
            print("feature name = " + f)
            # descriptive stats
            chk.loc[i, "train_min"] = np.nanmin(self.train[f].values)
            chk.loc[i, "test_min"] = np.nanmin(self.test[f].values)
            chk.loc[i, "train_mean"] = np.nanmean(self.train[f].values)
            chk.loc[i, "test_mean"] = np.nanmean(self.test[f].values)
            chk.loc[i, "train_max"] = np.nanmax(self.train[f].values)
            chk.loc[i, "test_max"] = np.nanmax(self.test[f].values)
            chk.loc[i, "train_skew"] = self.train.loc[self.train[f].isna().values == False, f].skew()
            chk.loc[i, "test_skew"] = self.test.loc[self.test[f].isna().values == False, f].skew()

            # nans
            chk.loc[i, "train_nans"] = self.train[f].isna().sum()
            chk.loc[i, "test_nans"] = self.test[f].isna().sum()

            # nuniques
            chk.loc[i, "train_nunique"] = self.train[f].nunique()
            chk.loc[i, "test_nunique"] = self.test[f].nunique()

        for c in ["train_nans", "test_nans", "train_nunique", "test_nunique"]:
            chk[c] = chk[c].astype(int)
        return chk

    # histogram for dense features
    def plot_bars(self):
        """
        plot histogram for each dense feature    
        """
        nrow, ncol = row_col(len(self.features))
        fig, ax = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4 * nrow))
        ax = ax.flatten()
        for i, f in enumerate(self.features):
            ax[i].hist(self.train.loc[self.train[f].isna().values == False, f], color="k", alpha=0.5)
            ax[i].hist(self.test.loc[self.test[f].isna().values == False, f], color="r", alpha=0.5)
            ax[i].axvline(np.nanmean(self.train[f].values), color="k", label="train")
            ax[i].axvline(np.nanmean(self.test[f].values), color="r", label="test")
            ax[i].legend(frameon=False)
            ax[i].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax[i].ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
            ax[i].set_title(f)
        for a in ax[len(self.features):]:
            a.axis("off")
        plt.tight_layout()

    # dense feature vs target
    def plot_vs_target(self):
        """
        plot vs target for each dense feature    
        """
        nrow, ncol = row_col(len(self.features))
        fig, ax = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4 * nrow))
        ax = ax.flatten()
        for i, f in enumerate(self.features):
            sns.regplot(x=f, y=self.target, data=self.train, ax=ax[i])
            ax[i].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax[i].ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
        for a in ax[len(self.features):]:
            a.axis("off")
        plt.tight_layout()
    
    # correlation matrix
    def correlation_matrix(self):
        X = self.train[self.features]

        # Compute the correlation matrix
        corr = X.corr()

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, cmap=cmap, center=0, annot=True, 
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        ax.set_ylim(corr.shape[0], 0)
        plt.yticks(rotation=0)