# original
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn

class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(600, 400),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(300, 100)
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, 300),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(300, 400),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(400, 600),
            nn.Sigmoid()
        )

    def forward(self, x):
        for l in self.encoder:
            x = l(x)
        for l in self.decoder:
            x = l(x)
        return x

model = Dense()
model.load_state_dict(torch.load("bottle.neck"), strict=False)
model.eval()

reg = pickle.load(open("LinearReg.pkl", 'rb'))

pr_scaler = pickle.load(open("pr_scaler.pkl", 'rb'))
vo_scaler = pickle.load(open("vo_scaler.pkl", 'rb'))

def get_r_hat(A, B): 
    """
        A: 1440-by-10 dataframe of log prices with columns log_pr_0, ... , log_pr_9
        B: 1440-by-10 dataframe of trading volumes with columns volu_0, ... , volu_9    
        return: a numpy array of length 10, corresponding to the predictions for the forward 30-minutes returns of assets 0, 1, 2, ..., 9
    """
    # X = A.iloc[-50:].values
    # X = np.concatenate([X, X[1:] - X[:-1]], axis=0)
    # pred = model.predict(X.reshape(1, -1))
    # print(A.shape)
    A = A.values.T
    B = B.values.T
    A = pr_scaler.transform(A[:,-30:].reshape(-1)[None,:])
    B = vo_scaler.transform(B[:,-30:].reshape(-1)[None,:])
    input = np.concatenate((A,B),axis = 1)
    input = torch.FloatTensor(input)
    output = model.encoder(input[None, :])
    # print(output.detach().numpy().shape)
    pred = reg.predict(output.detach().numpy()[0])
    # print(pred.shape)
    return pred - A[:,-1]
    # return np.array([m.predict(input)[0] for m in model]) - A[:,-1]
