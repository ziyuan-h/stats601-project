# original
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import pickle

<<<<<<< Updated upstream
with open('./kreg.pkl', 'rb') as f:
    model = pickle.load(f)

def get_r_hat(A, B): 
    """
        A: 1440-by-10 dataframe of log prices with columns log_pr_0, ... , log_pr_9
        B: 1440-by-10 dataframe of trading volumes with columns volu_0, ... , volu_9    
        return: a numpy array of length 10, corresponding to the predictions for the forward 30-minutes returns of assets 0, 1, 2, ..., 9
    """
    X = A.iloc[-50:].values
    X = np.concatenate([X, X[1:] - X[:-1]], axis=0)
    pred = model.predict(X.reshape(1, -1))
    return pred # Use the negative 30-minutes backward log-returns to predict the 30-minutes forward log-returns
    
=======
     return np.array([m.predict(input)[0] for m in model]) - A[:,-1]
>>>>>>> Stashed changes
