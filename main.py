import numpy as np
import pandas as pd
import pickle

with open('./linreg_fs_mi.pkl', 'rb') as f:
    file_linreg_fs_mi = pickle.load(f)

with open('./support_fs_mi.pkl', 'rb') as f:
    support_fs_mi = pickle.load(f)
    
def transform_split(X):
    prices = X[:,0,[-30,-1],:]
    price_diff = (X[:,0,1:,:] - X[:,0,:-1,:])[:,-30:,:]
    volu = np.log(X[:,1,:,:] + 1)[:,[-30, -1],:]
    volu_diff = (np.log(X[:,1,1:,:] + 1) - np.log(X[:,1,:-1,:] + 1))[:,-30:,:]
    
    X_train = np.concatenate([prices, price_diff, volu, volu_diff], axis=1)
    X_train_off49 = np.delete(X_train, [4, 9], axis=2)
    X_train_4 = np.concatenate([X_train[:,:,4], 
                                price_diff[:,:,[1,2,6,7]].sum(axis=1),
                                volu_diff[:,:,[1,2,6,7]].sum(axis=1),
                                prices[:,:,[1,2,6,7]].reshape(len(X_train),-1),
                                volu[:,:,[1,2,6,7]].reshape(len(X_train),-1)], axis=1)
    X_train_9 = np.concatenate([X_train[:,:,9],
                               X_train[:,:,:-1].reshape(len(X_train), -1)], axis=1)
    X_train_final = [X_train_off49[:,:,i] for i in range(8)]
    X_train_final.insert(4, X_train_4)
    X_train_final.insert(9, X_train_9)
    return X_train_final

# split models
def get_r_hat(A, B):
    X = np.stack([A.values, B.values], axis=0)
    X_test = transform_split(X[np.newaxis,:])
    pred = []
    for i in range(10):
        pred.append(file_linreg_fs_mi[i].predict(X_test[i][:,support_fs_mi[i]]))

    return pred
