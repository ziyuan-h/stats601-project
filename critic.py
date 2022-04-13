import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import time

class Critic:
    def __init__(self):
        pass


    # pair wise correlation of each asset, input and outputs are both numpy arrays
    def pairwise_corr(self, y_hat:np.array, y:np.array):
        n, p = y_hat.shape
        y_hat_pd = pd.DataFrame(y_hat, index=np.arange(n), columns=np.arange(p))
        y_pd = pd.DataFrame(y, index=np.arange(n), columns=np.arange(p))
        corr = y_pd.corrwith(y_hat_pd)
        return corr.values


    # overall correlation, input and outputs are both numpy arrays
    def overall_corr(self, y_hat:np.array, y:np.array):
        return np.corrcoef(y_hat.ravel(), y.ravel())[0,1]


    # scoring function
    def score(self, get_r_hat, log_pr, volu):
        """
        Calculate the correlation between the estimate value and true
        value of each asset's return

        Parameters:     
            get_r_hat(Callable): The function to be criticed
            log_pr(pd.DataFrame): The minutely log price of the period
                to be estimated
            volu(pd.DataFrame): The trading volume of the period to be 
                estimated

        Return:
            (float, pd.DataFrame, float): time used, pairwise correlation, and 
                overall correlation
        """
        # calculate r_hat over the test data range log_pr & volu
        t0 = time.time()
        dt = datetime.timedelta(days=1) - datetime.timedelta(minutes=1)
        r_hat = pd.DataFrame(index=log_pr.index[1440::10], 
                            columns=np.arange(10), 
                            dtype=np.float64)
        for t in log_pr.index[1440::10]: # compute the predictions every day
            r_hat.loc[t, :] = get_r_hat(log_pr.loc[(t - dt):t], volu.loc[(t - dt):t])

        t_used = time.time() - t0

        # Compute true forward log_returns every 10 minutes
        r_fwd = (log_pr.shift(-30) - log_pr).iloc[1440::10].rename(
                                            columns={f"log_pr_{i}": i for i in range(10)})

        pair_wise_corr = r_fwd.corrwith(r_hat)

        # Overall correlation (The ranking is based on this metric on the testing dataset)
        r_fwd_all = r_fwd.iloc[:-3].values.ravel() # the final 3 rows are NaNs. 
        r_hat_all = r_hat.iloc[:-3].values.ravel()
        
        overall_corr = np.corrcoef(r_fwd_all, r_hat_all)[0, 1]
        # print("overall correlation", overall_corr)

        return t_used, pair_wise_corr, overall_corr


    # submit a function to critic, return the analysis report
    def submit(self, get_r_hat, log_pr, volu):
        t_used, pairwise, overall = self.score(get_r_hat, log_pr, volu)
        print("Total time used: %.3fs" % t_used)
        pairwise_report = "Pairwise correlation:\n"
        for i in range(10):
            pairwise_report += "\tasset %d = %.5f\n" % (i, pairwise[i])
        pairwise_report += "\tmean correlation = %.5f" % pairwise.values.mean()
        print(pairwise_report)
        print("Overall correlation: %.5f" % overall)
        print("===============================")
        if pairwise.values.mean() > 0.02840 and overall > 0.01536:
            print("Performance beats Ziwei's dummy method!")
        else:
            print("Fail to outperform Ziwei's method, whose pairwise average\n" +
                    "and overall correlations are (%.5f, %.5f)" % (0.0284, 0.01536))
        print("===============================")
        return t_used, pairwise, overall
