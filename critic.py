import pandas as pd
import numpy as np
import datetime
import time

# Another (allegedly worse) example of get_r_hat / baseline function 2: pure random

def random_r_hat(A, B):
    return np.random.randn(10)

# An example of get_r_hat / baseline function 1: last-30-minutes log price prediction

def get_r_hat(A, B): 
    """
        A: 1440-by-10 dataframe of log prices with columns log_pr_0, ... , log_pr_9
        B: 1440-by-10 dataframe of trading volumes with columns volu_0, ... , volu_9    
        return: a numpy array of length 10, corresponding to the predictions for the forward 30-minutes returns of assets 0, 1, 2, ..., 9
    """
    
    return -(A.iloc[-1] - A.iloc[-30]).values # Use the negative 30-minutes backward log-returns to predict the 30-minutes forward log-returns
    
# validate functions

# pairwise correlation between the prediction and true value
def pairwise_corr(return_hat:pd.DataFrame, return_true:pd.DataFrame) -> float:
    """
    Parameters:
        return_hat(pd.DataFrame): n-by-10 dataframe for the prediction
        return_true(pd.DataFrame): n-by-10 dataframe for the true values
    """
    pair_wise_corr_ = return_true.corrwith(return_hat).to_numpy()
    return np.linalg.norm(pair_wise_corr_ - np.ones_like(pair_wise_corr_), ord=1)

# overall correlation between the prediction and true value
def overall_corr(return_hat:pd.DataFrame, return_true:pd.DataFrame) -> float:
    """
    Parameters:
        return_hat(pd.DataFrame): n-by-10 dataframe for the prediction
        return_true(pd.DataFrame): n-by-10 dataframe for the true values
    """
    return_hat_all = return_hat.values.ravel()
    return_true_all = return_true.values.ravel()
    return np.corrcoef(return_hat_all, return_true_all)

def corr_score(get_r_hat, log_pr:pd.DataFrame, volu:pd.DataFrame) -> tuple:
    """
    Calculate the correlation between the estimate value and true
    value of each asset's return

    Parameters:     
        get_r_hat(Callable): The function to be criticed
        log_pr(pd.DataFrame): The minutely log price of the period(one day) 
            to be estimated
        volu(pd.DataFrame): The trading volume of the period(one day) to be 
            estimated

    Return:
        (float, pd.DataFrame, float): time used, pairwise correlation, and 
            overall correlation
    """
    # calculate r_hat over the test data range log_pr & volu
    t0 = time.time()
    dt = datetime.timedelta(days=1) - datetime.timedelta(minutes=1)
    r_hat = pd.DataFrame(index=log_pr.index[1440::1440], 
                        columns=np.arange(10), 
                        dtype=np.float64)
    for t in log_pr.index[1440::1440]: # compute the predictions every day
        # print(log_pr.loc[(t - dt):t].shape)
        r_hat.loc[t, :] = get_r_hat(log_pr.loc[(t - dt):t], volu.loc[(t - dt):t])
        # print("calculating", t)
    t_used = time.time() - t0
    # print("time used to calculate r_hat", t_used)

    # Compute true forward log_returns every 10 minutes
    r_fwd = (log_pr.shift(-30) - log_pr).iloc[1440::1440].rename(
                                        columns={f"log_pr_{i}": i for i in range(10)})

    # print correlation with every asset
    pair_wise_corr = r_fwd.corrwith(r_hat)
    # print("correlation with every asset\n", pair_wise_corr)

    # Overall correlation (The ranking is based on this metric on the testing dataset)
    r_fwd_all = r_fwd.iloc[:-3].values.ravel() # the final 3 rows are NaNs. 
    r_hat_all = r_hat.iloc[:-3].values.ravel()
    
    overall_corr = np.corrcoef(r_fwd_all, r_hat_all)[0, 1]
    # print("overall correlation", overall_corr)

    return t_used, pair_wise_corr, overall_corr
