import numpy as np
import pandas as pd

def get_r_hat(A, B): 
    """
        A: 1440-by-10 dataframe of log prices with columns log_pr_0, ... , log_pr_9
        B: 1440-by-10 dataframe of trading volumes with columns volu_0, ... , volu_9    
        return: a numpy array of length 10, corresponding to the predictions for the forward 30-minutes returns of assets 0, 1, 2, ..., 9
    """
    
    data = pd.concat([A, B], axis=1).values.ravel()[np.newaxis, :]
    weights = np.loadtxt("./linreg.txt", delimiter=',')
    pred = weights[0] + (data @ weights[1:]).squeeze()
    return A.iloc[-1].values - pred # Use the negative 30-minutes backward log-returns to predict the 30-minutes forward log-returns
    