# %%
import numpy as np
import pandas as pd

log_pr_file = './log_price.df'
volu_usd_file = './volume_usd.df'

log_pr = pd.read_pickle(log_pr_file)
volu = pd.read_pickle(volu_usd_file)

daylen = 10

def interpolate(log_pr, volu, window=30):
    log_pr.columns = ['log_pr_%d'%i for i in range(10)]
    volu.columns = ['volu_%d'%i for i in range(10)]

    open_ = log_pr[::window].reindex(log_pr.index).ffill()
    open_.columns = ['open_%d'%i for i in range(10)]
    close_ = log_pr[window-1::window].reindex(log_pr.index).bfill()
    close_.columns = ['close_%d'%i for i in range(10)]
    high_ = log_pr.groupby(np.arange(len(log_pr))//window) \
            .max().set_index(np.arange(0, len(log_pr), window)) \
            .reindex(np.arange(len(log_pr))).ffill().set_index(log_pr.index)
    high_.columns = ['high_%d'%i for i in range(10)]
    low_ = log_pr.groupby(np.arange(len(log_pr))//window) \
            .min().set_index(np.arange(0, len(log_pr), window)) \
            .reindex(np.arange(len(log_pr))).ffill().set_index(log_pr.index)
    low_.columns = ['low_%d'%i for i in range(10)]
    return pd.concat([log_pr, volu, open_, close_, high_, low_], axis=1)

# data = interpolate(log_pr, volu, daylen)

# %%
# Simple Moving Average
def SMA(x, window):
    return x.rolling(window).mean()

# exponential moving average
def EMA(x, window):
    return x.ewm(com=1/window, adjust=True, min_periods=window).mean()

# Average True Range
def ATR(x, window, daylen):
    low = x[['low_%d'%i for i in range(10)]].iloc[::daylen].copy()
    high = x[['high_%d'%i for i in range(10)]].iloc[::daylen].copy()
    close = x[['close_%d'%i for i in range(10)]].iloc[::daylen].copy()
    
    high_low = high.values - low.values
    high_close = np.abs(high.values - close.shift().values)
    low_close = np.abs(low.values - close.shift().values)

    ranges = np.stack([high_low, high_close, low_close], axis=0)
    true_range = np.max(ranges, axis=0)
    true_range = pd.DataFrame(true_range, 
                              index=close.index, columns=['atr_%d'%i for i in range(10)])
    atr = EMA(true_range, window)
    atr = atr.reindex(x.index).ffill()
    return atr

# TODO
# Average Directional Movement Index
def ADX(x, window, daylen):
    low = x[['low_%d'%i for i in range(10)]].iloc[::daylen].copy()
    high = x[['high_%d'%i for i in range(10)]].iloc[::daylen].copy()
    close = x[['close_%d'%i for i in range(10)]].iloc[::daylen].copy()
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    atr = ATR(x, window, daylen).iloc[::daylen]
#     print(atr)
    
    plus_di = (100 * EMA(plus_dm, window) / atr.values).values
    minus_di = abs(100 * EMA(minus_dm, window) / atr.values).values
    
    adx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = pd.DataFrame(adx, index=close.index, columns=['adx_%d'%i for i in range(10)])
    adx = ((adx.shift() * (window - 1)) + adx) / window
    adx_smooth = EMA(adx, window)
    adx_smooth = adx_smooth.reindex(x.index).ffill()
    return adx_smooth

# Commodity Channel Index
def CCI(x, window, daylen):
    low = x[['low_%d'%i for i in range(10)]].iloc[::daylen].copy()
    high = x[['high_%d'%i for i in range(10)]].iloc[::daylen].copy()
    close = x[['close_%d'%i for i in range(10)]].iloc[::daylen].copy()
    
    m = (high.values + low.values + close)/3
#     return m
    sma = SMA(m, window)
#     return sma
    mad_ = m.rolling(window).apply(lambda x: pd.Series(x).mad())
    cci = pd.DataFrame((m.values - sma.values)/(0.015*mad_.values), 
                       index=close.index, columns=['cci_%d'%i for i in range(10)])
    cci = cci.reindex(x.index).ffill()
    return cci

# Price Rate of Change
def ROC(x, window, daylen):
    close = x[['close_%d'%i for i in range(10)]].iloc[::daylen].copy()
    roc = close.pct_change(window)
    roc.columns = ['roc_%d'%i for i in range(10)]
    roc = roc.reindex(x.index).ffill()
    return roc

# Relative Strength Index
def RSI(x, window, daylen, ema=True):
    close = x[['close_%d'%i for i in range(10)]].iloc[::daylen].copy()
    
    # Make two series: one for lower closes and one for higher closes
    up = close.diff().clip(lower=0)
    down = -1 * close.diff().clip(upper=0)
    
    if ema == True:
        # Use exponential moving average
        ma_up = EMA(up, window)
        ma_down = EMA(down, window)
    else:
        # Use simple moving average
        ma_up = SMA(up, window)
        ma_down = SMA(down, window)
        
    rsi = ma_up.values / (ma_down.values + 1e-4)
    rsi = 100 - (100/(1 + rsi))
    rsi = pd.DataFrame(rsi, index=close.index, columns=['rsi_%d'%i for i in range(10)])
    rsi = rsi.reindex(x.index).ffill()
    return rsi

# William's %R oscillator
def WR(x, window):
    hn = x[['log_pr_%d'%i for i in range(10)]].rolling(window).max()
    ln = x[['log_pr_%d'%i for i in range(10)]].rolling(window).min()
    wr = 100*(hn.values - x[['close_%d'%i for i in range(10)]].values)/(hn.values - ln.values)
    return pd.DataFrame(wr, index=x.index, columns=['wr_%d'%i for i in range(10)])

# Stochastic K
def SK(x, window):
    hhn = x[['high_%d'%i for i in range(10)]].rolling(window).max()
    lln = x[['low_%d'%i for i in range(10)]].rolling(window).min()
    sk = 100*(x[['close_%d'%i for i in range(10)]].values - lln.values)/(hhn.values - lln.values)
    return pd.DataFrame(sk, index=x.index, columns=['sk_%d'%i for i in range(10)])

# Stochastic D
def SD(x, window):
    sd = EMA(SK(x, window), 3)
    sd.columns = ['sd_%d'%i for i in range(10)]
    return sd
    

# %%
# feature generation pipline
def generate_features(data, window, daylen):
    pr = data.drop(labels=['volu_%d'%i for i in range(10)], axis=1)
    sma = SMA(pr[['log_pr_%d'%i for i in range(10)]], window)
    sma.columns = ['sma_%d'%i for i in range(10)]
    # print(sma.shape)
    ema = EMA(pr[['log_pr_%d'%i for i in range(10)]], window)
    ema.columns = ['ema_%d'%i for i in range(10)]
    # print(ema.shape)
    atr = ATR(pr, window, daylen)
    # print(atr.shape)
    adx = ADX(pr, window, daylen)
    # print(adx.shape)
    # cci = CCI(pr, window, daylen)
    # print(cci.shape)
    # roc = ROC(pr, window, daylen)
    # print(roc.shape)
    rsi = RSI(pr, window, daylen)
    # print(rsi.shape)
    wr = WR(pr, window)
    # print(wr.shape)
    sk = SK(pr, window)
    # print(sk.shape)
    sd = SD(pr, window)
    # print(sd.shape)
    return pd.concat([sma, ema, atr, adx, # cci, roc, 
                    rsi, wr, sk, sd], axis=1)


# %%
# combined pipeline
from sklearn.preprocessing import StandardScaler
import pickle

def data_preprocess(log_pr, volu, window, daylen, scaler_file = None):
    data = interpolate(log_pr, volu, window)
    features = generate_features(data, window, daylen)
    # print(features.shape)
    features = features.dropna()
    # print(features.shape)
    if isinstance(scaler_file, type(None)):
        scaler = StandardScaler()
        features_transform = scaler.fit_transform(features)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    else:
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        features_transform = scaler.transform(features)
    features_transform = pd.DataFrame(features_transform, 
                                    index=features.index, 
                                    columns=features.columns)
    return features_transform, scaler_file

