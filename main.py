import numpy as np
import pandas as pd
import pickle

# An example of get_r_hat

def get_r_hat(A, B):
    ma = A.rolling(11).mean()
    return ma.values[-1] - A.values[-1]