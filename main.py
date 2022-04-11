import numpy as np
from joblib import load
import pandas as pd
from sklearn.preprocessing import normalize
import pickle
import os
import random
from string import ascii_letters
import torch
from torch import nn
import torch.nn.functional as F


import torch.nn as nn
import torch.nn.functional as F

class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()
        self.lin1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(2880, 1024)
        self.lin3 = nn.Linear(1024, 256)
        self.lin4 = nn.Linear(256, 1)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.2)
    
    def forward(self, x):
        x = self.lin1(x) # b x 4880 x 10
        x = self.relu(x) # b x 4880 x 10
        x = x.transpose(1,2) # b x 10 x 4880
        x  = self.drop1(x)
        x = self.lin2(x) # b x 10 x 1024
        x = self.relu(x) # b x 10 x 1024
        x = self.drop2(x)
        x = self.lin3(x) # b x 10 x 256
        x = self.relu(x) # b x 10 x 256
        x = self.drop3(x)
        x = self.lin4(x) # b x 10 x 1
        x = x.squeeze(-1) # b x 10
        return x
    
net = Dense()


net.load_state_dict(torch.load('./NN.mdl'), strict=False)
def get_r_hat(A, B): 
    """
        A: 1440-by-10 dataframe of log prices with columns log_pr_0, ... , log_pr_9
        B: 1440-by-10 dataframe of trading volumes with columns volu_0, ... , volu_9    
        return: a numpy array of length 10, corresponding to the predictions for the forward 30-minutes returns of assets 0, 1, 2, ..., 9
    """
    data = np.vstack([A.values, B.values])
    return net(torch.tensor(data).unsqueeze(0).float()).detach().numpy()
        