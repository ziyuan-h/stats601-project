import os
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit

# !pip3 install bayesian-optimization
# import hyperparam
import critic

def FastAllFeatureExtract(A):
    A0_pr = A[:, :1440]
    A0_vo = A[:, 1440:] + 1
    feature = A0_pr[:, -1] - A0_pr[:, 0]
    # VO volumn log################
    feature = np.concatenate((feature[:, None], np.log(A0_vo[:, -30:])), axis=1)

    # VO moving avg (#sample,91:120)########
    avg_step = 30
    df_vo = pd.DataFrame(A0_vo[:, -59:].T)
    ma_30 = lambda x: x.rolling(avg_step).mean()
    df_vo.apply(ma_30).apply(np.log).T.to_numpy()[:, avg_step - 1:]
    feature = np.append(feature, df_vo.apply(ma_30).apply(np.log).T.to_numpy()[:, -30:], axis=1)

    # PR rate of change (#sample, 271:300)#######
    df_pr = pd.DataFrame(A0_pr[:, -31:].T)
    pct_chg_fxn = lambda x: x.pct_change()
    # print("PR rate of change", df_pr.apply(pct_chg_fxn).T.to_numpy()[:,-30:])
    feature = np.append(feature, df_pr.apply(pct_chg_fxn).T.to_numpy()[:, -30:], axis=1)

    # PR moving avg (#sample,301:330)  #######
    df_pr = pd.DataFrame(A0_pr[:, -59:].T)
    avg_step = 30
    ma_30 = lambda x: x.rolling(avg_step).mean()
    df_pr.apply(ma_30).T.to_numpy()[:, avg_step - 1:]
    # print("PR moving avg", df_pr.apply(ma_30).T.to_numpy()[:,-30:])
    feature = np.append(feature, df_pr.apply(ma_30).T.to_numpy()[:, -30:], axis=1)

    # PR binning (#sample, 331:360)#########
    df_pr = pd.DataFrame(A0_pr[:, -30:].T)
    n_bins = 10
    bin_fxn = lambda y: pd.qcut(y, q=n_bins, labels=range(1, n_bins + 1))
    binning = df_pr.apply(bin_fxn).T
    # print("PR binning", binning.to_numpy()[:,-30:])
    feature = np.append(feature, binning.to_numpy(), axis=1)

    return feature