# normalize
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer

reg_weights = np.loadtxt("./linreg_norm.txt", delimiter=',')

def get_r_hat(A, B):
    """
        A: 1440-by-10 dataframe of log prices with columns log_pr_0, ... , log_pr_9
        B: 1440-by-10 dataframe of trading volumes with columns volu_0, ... , volu_9    
        return: a numpy array of length 10, corresponding to the predictions for the forward 30-minutes returns of assets 0, 1, 2, ..., 9
    """
    A_np, B_np = A.values, B.values
    data = np.concatenate([A_np, np.log(B_np + 1)], axis=0).reshape(1, -1)
    data = Normalizer().fit_transform(data)
    predict = (data @ reg_weights[1:]).squeeze() + reg_weights[0]
    return predict
