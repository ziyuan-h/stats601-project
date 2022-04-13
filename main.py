import numpy as np
from joblib import load
import pandas as pd
from sklearn.preprocessing import normalize
import pickle
import os
import random


import pickle

def FastAllFeatureExtract(A):
    A0_pr = A[:,:1440]
    A0_vo = A[:,1440:]+1
    feature  = A0_pr[:,-1] - A0_pr[:,0]
    #VO volumn log################
    feature = np.concatenate((feature[:,None], np.log(A0_vo[:,-30:])), axis = 1)
       
    #VO moving avg (#sample,91:120)########
    avg_step = 30
    df_vo = pd.DataFrame(A0_vo[:,-59:].T)
    ma_30 = lambda x: x.rolling(avg_step).mean()
    df_vo.apply(ma_30).apply(np.log).T.to_numpy()[:,avg_step-1:]
    feature = np.append(feature,df_vo.apply(ma_30).apply(np.log).T.to_numpy()[:,-30:], axis = 1)
    
    #PR rate of change (#sample, 271:300)#######
    df_pr = pd.DataFrame(A0_pr[:,-31:].T)
    pct_chg_fxn = lambda x: x.pct_change()
    #print("PR rate of change", df_pr.apply(pct_chg_fxn).T.to_numpy()[:,-30:])
    feature = np.append(feature,df_pr.apply(pct_chg_fxn).T.to_numpy()[:,-30:], axis = 1)
    
    #PR moving avg (#sample,301:330)  #######
    df_pr = pd.DataFrame(A0_pr[:,-59:].T)
    avg_step = 30
    ma_30 = lambda x: x.rolling(avg_step).mean()
    df_pr.apply(ma_30).T.to_numpy()[:,avg_step-1:]
    #print("PR moving avg", df_pr.apply(ma_30).T.to_numpy()[:,-30:])
    feature = np.append(feature,df_pr.apply(ma_30).T.to_numpy()[:,-30:], axis = 1) 
    
    #PR binning (#sample, 331:360)#########
    df_pr = pd.DataFrame(A0_pr[:,-30:].T)
    n_bins = 10
    bin_fxn = lambda y: pd.qcut(y,q=n_bins,labels = range(1,n_bins+1))
    binning = df_pr.apply(bin_fxn).T
    #print("PR binning", binning.to_numpy()[:,-30:])
    feature = np.append(feature,binning.to_numpy(), axis = 1)
    
    return feature

ass0 = pickle.load(open('LR0.plk', 'rb'))
ass1 = pickle.load(open('LR1.plk', 'rb'))
ass2 = pickle.load(open('LR2.plk', 'rb'))
ass3 = pickle.load(open('LR3.plk', 'rb'))
ass4 = pickle.load(open('LR4.plk', 'rb'))
ass5 = pickle.load(open('LR5.plk', 'rb'))
ass6 = pickle.load(open('LR6.plk', 'rb'))
ass7 = pickle.load(open('LR7.plk', 'rb'))
ass8 = pickle.load(open('LR8.plk', 'rb'))
ass9 = pickle.load(open('LR9.plk', 'rb'))

def get_r_hat(A, B):
    A = A.values.T
    B = B.values.T
    iin =np.concatenate((A,B), axis=1)
    mid = FastAllFeatureExtract(iin)
    in0 = mid[[0],:]
    in1 = mid[[1],:]
    diff_vol = B[[2],-30:] - B[[2],-60:-30]
    in2 = np.concatenate((mid[[2],:31],diff_vol,mid[[2],31:121]),axis = 1)
    in3 = mid[[3],:]
    in4 = mid[[4],:]
    in5 = mid[[5],:]
    in6 = mid[[6],:]
    in7 = mid[[7],list(range(0, 1))+ list(range(31, 151))][None,:]
    in8 = mid[[8],list(range(0, 1))+ list(range(61, 151))][None,:]
    in9 = mid[[9],list(range(0, 1))+ list(range(61, 151))][None,:]

    return np.array([ass0.predict(in0)[0],ass1.predict(in1)[0],ass2.predict(in1)[0],\
         ass3.predict(in3)[0],ass4.predict(in4)[0],ass5.predict(in5)[0],\
          ass6.predict(in6)[0],ass7.predict(in7)[0],ass8.predict(in8)[0],\
         ass9.predict(in9)[0]]) - A[:,-1]