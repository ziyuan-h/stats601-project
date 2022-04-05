import numpy as np
from joblib import load
import pandas as pd
from sklearn.preprocessing import normalize
import pickle

def get_r_hat(A, B): 
    """
        A: 1440-by-10 dataframe of log prices with columns log_pr_0, ... , log_pr_9
        B: 1440-by-10 dataframe of trading volumes with columns volu_0, ... , volu_9    
        return: a numpy array of length 10, corresponding to the predictions for the forward 30-minutes returns of assets 0, 1, 2, ..., 9
    """
    result = np.zeros((10,1))
    
    X = np.concatenate((A.to_numpy(),B.to_numpy()),axis = 0)
    print(X[:,0].shape)
    for i in range(10):
        clf = pickle.load(open(f'model{i}.pkl','rb') )
        result[i,0] = clf.predict(np.reshape(X[:,i],(1,2880)))
    # print(result / np.linalg.norm(result))
    return pd.DataFrame(result / np.linalg.norm(result)).values
        