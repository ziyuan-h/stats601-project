import pandas as pd
import numpy as np

class DataSet:
    def __init__(self, log_price_file = "./log_price.df", volume_file="./volume_usd.df", 
                        test_size=1440 * 60):
        self.log_pr = pd.read_pickle(log_price_file)
        self.volu = pd.read_pickle(volume_file)
        self.formulized = False
        self.parse(test_size=test_size)


    # parse data into training set and validation set
    def parse(self, test_size = 1440 * 60):
        self.log_pr_train = self.log_pr.iloc[:-test_size]
        self.volu_train = self.volu.iloc[:-test_size]
        self.log_pr_test = self.log_pr.iloc[-test_size:]
        self.volu_test = self.volu.iloc[-test_size:]


    # formulize the training data into the shape of (n_samples, 2, features, 10)
    def formulize(self, feature_size = 1440, step_size = 10):
        """
        Input parameters:
            feature_size: the number of data used in a feature vector
            window_size: frequency of sampling the feature vector
        """
        self.formulized = True
        log_pr_train_np, volu_train_np = self.log_pr_train.values, self.volu_train.values
        index = np.arange(0, feature_size)[np.newaxis, :] + \
                step_size * np.arange(0, (len(log_pr_train_np) - feature_size - 30) // step_size)[:, np.newaxis]
        label_index = index[:,-1] + 30
        # print(index[:10])
        self.train_f = np.stack([log_pr_train_np[index], volu_train_np[index]], axis=1)
        self.label_f = log_pr_train_np[label_index].squeeze()
        return self.train_f.shape, self.label_f.shape


    @property
    def train_set(self):
        return [self.log_pr_train, self.volu_train]


    @property
    def test_set(self):
        return [self.log_pr_test, self.volu_test]


    @property
    def train_set_form(self):
        if not self.formulized:
            self.formulize()
        return self.train_f, self.label_f


    def train_set_log_return(self, diff = 1):
        return (self.log_pr_train.shift(-diff) - self.log_pr_train).dropna()