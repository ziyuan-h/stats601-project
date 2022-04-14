import pandas as pd
import numpy as np

class DataSet:
    def __init__(self, log_price_file = "./log_price.df", volume_file="./volume_usd.df", 
                        test_size=1440 * 60, step_size=10):
        self.log_pr = pd.read_pickle(log_price_file)
        self.volu = pd.read_pickle(volume_file)
        self.formulized_plane = False
        self.formulized_shuffle = False
        self.parse(test_size=test_size)
        self.step_size = step_size
        self.test_size = test_size


    # parse data into training set and validation set
    def parse(self, test_size = 1440 * 60):
        self.log_pr_train = self.log_pr.iloc[:-test_size]
        self.volu_train = self.volu.iloc[:-test_size]
        self.log_pr_test = self.log_pr.iloc[-test_size:]
        self.volu_test = self.volu.iloc[-test_size:]
        
    def _formulize_helper(self, log_pr_train_np, volu_train_np, feature_size, step_size):
        index = np.arange(0, feature_size)[np.newaxis, :] + \
                step_size * np.arange(0, (len(log_pr_train_np) - feature_size - 30) // step_size)[:, np.newaxis]
        label_index = index[:,-1] + 30
        # print(index[:10])
        train_f = np.stack([log_pr_train_np[index], volu_train_np[index]], axis=1)
        label_f = log_pr_train_np[label_index].squeeze()
        return train_f, label_f


    # formulize the training data into the shape of (n_samples, 2, features, 10)
    def formulize_plane(self, feature_size = 1440):
        """
        Input parameters:
            feature_size: the number of data used in a feature vector
            window_size: frequency of sampling the feature vector
        """
        self.formulized_plane = True
        log_pr_train_np, volu_train_np = self.log_pr_train.values, self.volu_train.values
        self.train_f, self.label_f = self._formulize_helper(log_pr_train_np, volu_train_np, feature_size, self.step_size)
        return self.train_f.shape, self.label_f.shape
    
   
    # don't use it as of now
    def formulize_shuffle(self, test_size, feature_size = 1440):
        self.formulized_shuffle = True
        log_pr_np, volu_np = self.log_pr.values, self.volu.values
        total_sf, label_sf = self._formulized_helper(log_pr_np, volu_np, feature_size, self.step_size)
        shuffle_index = np.random.shuffle(np.arange(len(self.total_sf)))
        self.train_sf, self.test_sf = total_sf[shuffle_index[:-test_size]], total_sf[shuffle_index[-test_size:]]
        self.train_label_sf, self.test_label_sf = label_sf[shuffle_index[:-test_sizse]], label_sf[shuffle_index[-test_size:]]
        return self.train_sf.shape, self.train_label_sf.shape, self.test_sf.shape, self.test_label_sf.shape
 

    @property
    def train_set(self):
        return [self.log_pr_train, self.volu_train]


    @property
    def test_set(self):
        return [self.log_pr_test, self.volu_test]


    @property
    def train_set_form(self):
        if not self.formulized_plane:
            self.formulize_plane()
        return self.train_f, self.label_f
    
    # don't use it as of now
    @property
    def train_set_form_shuffle(self):
        if not self.formulized:
            self.formulize()
        return self.train_sf, self.train_label.sf
    
    
    @property
    def submit_train(self):
        self.total = self._formulize_helper(self.log_pr.values,
                                            self.volu.values, 
                                            1440, 10)
        return self.total


    def train_set_log_return(self, diff = 1):
        return (self.log_pr_train.shift(-diff) - self.log_pr_train).dropna()