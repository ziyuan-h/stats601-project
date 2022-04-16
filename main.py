import os
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
import random
from string import ascii_letters
import torch
from torch import nn
import torch.nn.functional as F

def FastAllFeatureExtract(A):
    d = 28
    A0_pr = A[:, :1440]
    A0_vo = A[:, 1440:] + 1
    feature = A0_pr[:, -1] - A0_pr[:, 0]
    # VO volumn log################
    feature = np.concatenate((feature[:, None], np.log(A0_vo[:, -30+d:])), axis=1)

    # VO moving avg (#sample,91:120)########
    avg_step = 30
    df_vo = pd.DataFrame(A0_vo[:, -59+d:].T)
    ma_30 = lambda x: x.rolling(avg_step).mean()
    df_vo.apply(ma_30).apply(np.log).T.to_numpy()[:, avg_step - 1:]
    feature = np.append(feature, df_vo.apply(ma_30).apply(np.log).T.to_numpy()[:, -30+d:], axis=1)

    # PR rate of change (#sample, 271:300)#######
    df_pr = pd.DataFrame(A0_pr[:, -31:].T)
    pct_chg_fxn = lambda x: x.pct_change()
    # print("PR rate of change", df_pr.apply(pct_chg_fxn).T.to_numpy()[:,-30:])
    feature = np.append(feature, df_pr.apply(pct_chg_fxn).T.to_numpy()[:, -30+d:], axis=1)

    # PR moving avg (#sample,301:330)  #######
    df_pr = pd.DataFrame(A0_pr[:, -59:].T)
    avg_step = 30
    ma_30 = lambda x: x.rolling(avg_step).mean()
    df_pr.apply(ma_30).T.to_numpy()[:, avg_step - 1:]
    # print("PR moving avg", df_pr.apply(ma_30).T.to_numpy()[:,-30:])
    feature = np.append(feature, df_pr.apply(ma_30).T.to_numpy()[:, -30+d:], axis=1)

    # PR binning (#sample, 331:360)#########
    df_pr = pd.DataFrame(A0_pr[:, -30+d:].T)
    n_bins = 10
    bin_fxn = lambda y: pd.qcut(y, q=n_bins, labels=range(1, n_bins + 1))
    binning = df_pr.apply(bin_fxn).T
    # print("PR binning", binning.to_numpy()[:,-30:])
    feature = np.append(feature, binning.to_numpy(), axis=1)

    return feature


class Rescale(nn.Module):
    def __init__(self):
        super(Rescale, self).__init__()
        # fc = []
        # for i in range(10):
        #     fc.append(nn.Linear(1, 1, bias = True))
        # self.fc = nn.ModuleList(fc)
        self.zzw = torch.FloatTensor(np.ones((10,1)))
        self.zzw.requires_grad_()


    def forward(self, x):
        #print((x[:,0])[:,None].shape)
        out = torch.empty((x.shape),dtype=torch.float)
        for i in range(x.shape[0]):
            for j in range(10):
            #print(x[i,:].shape)
                out[i,j] = x[i,j] * self.zzw[j] *self.zzw[j]
        return out
        #return torch.FloatTensor([self.fc[i]((x[:,i])[:,None]) for i in range(10)])



final_model = Rescale()
final_model.load_state_dict(torch.load("final_scale.mdl"))
final_model.eval()

from pickle import load
linear_models = []
for i in range(10):
    linear_models.append(load(open(f'{i}.FFXGB','rb')))

def get_r_hat(A,B):
    A = A.values.T
    B = B.values.T
    features = FastAllFeatureExtract(np.concatenate((A,B), axis = 1))
    window = 20
    def getNpredict(i):
        filtered = np.zeros(40,dtype=complex)
        raw_fft = np.fft.fft(A[i,:])
        filtered[:window] = raw_fft[:window]
        filtered[-window:] = raw_fft[-window:]
        hi = np.append(filtered.real,filtered.imag)[None,:]
        hi = np.concatenate((hi,features[[i],:]),axis = 1)
        return linear_models[i].predict(hi)[0]- A[i,-1]
    #before_scale =  torch.FloatTensor([getNpredict(i) for i in range(10)] - A[:,-1])
    #after_scale =  final_model(before_scale[None,:])
    return -np.array([getNpredict(i) for i in range(10)]) *  np.array([3,8,4,8,4,1,8,4,10,9])
    #return after_scale.detach().numpy()


    