# %%
from pickle import FALSE
import numpy as np
import pandas as pd
# import ojsim

# sim = ojsim.OJSimulator()
# log_pr_train, volu_train = sim.train
# X, y = sim.formulized_train

class TechnicalAnalysis:
    def __init__(self, price, volu, window_size=10, fillna=False) -> None:
        self.set_params(price, volu, window_size, fillna)

    def set_params(self, price, volu, window_size, fillna):
        self.log_pr_train = price
        self.volume = volu
        self.window_size = window_size
        self.fillna = fillna
        
    def train_feature(self):
        # %%
        features_list = []

        # %% [markdown]
        # # Technical Analysis

        # %%
        # parameter list
        # window_size = 1

        # %% [markdown]
        # ## Momentun

        # %%
        import ta.momentum as momentum

        # %% [markdown]
        # **Kaufman’s Adaptive Moving Average (KAMA)**
        # 
        # Moving average designed to account for market noise or volatility. KAMA will closely follow prices when the price swings are relatively small and the noise is low. KAMA will adjust when the price swings widen and follow prices from a greater distance. This trend-following indicator can be used to identify the overall trend, time turning points and filter price movements.

        # %%
        for i in range(10):
            ds = momentum.kama(self.log_pr_train[i], self.window_size, fillna=self.fillna)
            ds.name = 'kama%d'%i
            features_list.append(ds)
        len(features_list)

        # %% [markdown]
        # **Percentage Price Oscillator (PPO)** is a momentum oscillator that measures the difference between two moving averages as a percentage of the larger moving average.

        # %%
        for i in range(10):
            ds = momentum.ppo(self.log_pr_train[i], window_fast=self.window_size, fillna=self.fillna)
            ds.name = 'ppo%d'%i
            features_list.append(ds)
        len(features_list)


        # %% [markdown]
        # **The Percentage Volume Oscillator (PVO)** is a momentum oscillator for volume. The PVO measures the difference between two volume-based moving averages as a percentage of the larger moving average.

        # %%
        for i in range(10):
            ds = momentum.pvo(self.volume[i], window_fast=self.window_size, fillna=self.fillna)
            ds.name = 'pvo%d'%i
            features_list.append(ds)
        len(features_list)

        # %% [markdown]
        # **Rate of Change (ROC)**
        # 
        # The Rate-of-Change (ROC) indicator, which is also referred to as simply Momentum, is a pure momentum oscillator that measures the percent change in price from one period to the next. The ROC calculation compares the current price with the price “n” periods ago. The plot forms an oscillator that fluctuates above and below the zero line as the Rate-of-Change moves from positive to negative. As a momentum oscillator, ROC signals include centerline crossovers, divergences and overbought-oversold readings. Divergences fail to foreshadow reversals more often than not, so this article will forgo a detailed discussion on them. Even though centerline crossovers are prone to whipsaw, especially short-term, these crossovers can be used to identify the overall trend. Identifying overbought or oversold extremes comes naturally to the Rate-of-Change oscillator.

        # %%
        for i in range(10):
            ds = momentum.roc(self.log_pr_train[i], window=self.window_size, fillna=self.fillna)
            ds.name = 'roc%d'%i
            features_list.append(ds)
        len(features_list)

        # %% [markdown]
        # **Stochastic RSI**
        # 
        # The StochRSI oscillator was developed to take advantage of both momentum indicators in order to create a more sensitive indicator that is attuned to a specific security’s historical performance rather than a generalized analysis of price change.
        # 
        # https://school.stockcharts.com/doku.php?id=technical_indicators:stochrsi https://www.investopedia.com/terms/s/stochrsi.asp

        # %%
        for i in range(10):
            ds = momentum.rsi(self.log_pr_train[i], window=self.window_size, fillna=self.fillna)
            ds.name = 'rsi%d'%i
            features_list.append(ds)
        len(features_list)

        # %% [markdown]
        # **self.fillna strength index (TSI)**
        # 
        # Shows both trend direction and overbought/oversold conditions.

        # %%
        for i in range(10):
            ds = momentum.tsi(self.log_pr_train[i], window_fast=self.window_size, fillna=self.fillna)
            ds.name = 'tsi%d'%i
            features_list.append(ds)
        len(features_list)

        # %% [markdown]
        # ## Volume

        # %%
        import ta.volume as tavolume

        # %% [markdown]
        # **Force Index (FI)**
        # 
        # It illustrates how strong the actual buying or selling pressure is. High positive values mean there is a strong rising trend, and low values signify a strong downward trend.
        # 
        # http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:force_index

        # %%
        for i in range(10):
            ds = tavolume.force_index(self.log_pr_train[i], self.volume[i], self.window_size, self.fillna)
            ds.name = 'fi%d'%i
            features_list.append(ds)
        len(features_list)

        # %% [markdown]
        # **Negative Volume Index (NVI)**
        # 
        # http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:negative_volume_inde

        # %%
        for i in range(10):
            ds = tavolume.negative_volume_index(self.log_pr_train[i], self.volume[i], self.fillna)
            ds.name = 'nvi%d'%i
            features_list.append(ds)
        len(features_list)

        # %% [markdown]
        # **On-balance volume (OBV)**
        # 
        # It relates price and volume in the stock market. OBV is based on a cumulative total volume.
        # 
        # https://en.wikipedia.org/wiki/On-balance_volume

        # %%
        for i in range(10):
            ds = tavolume.on_balance_volume(self.log_pr_train[i], self.volume[i], self.fillna)
            ds.name = 'obv%d'%i
            features_list.append(ds)
        len(features_list)

        # %% [markdown]
        # **Volume-price trend (VPT)**
        # 
        # Is based on a running cumulative volume that adds or substracts a multiple of the percentage change in share price trend and current volume, depending upon the investment’s upward or downward movements.
        # 
        # https://en.wikipedia.org/wiki/Volume%E2%80%93price_trend

        # %%
        for i in range(10):
            ds = tavolume.volume_price_trend(self.log_pr_train[i], self.volume[i], self.fillna)
            ds.name = 'vpt%d'%i
            features_list.append(ds)
        len(features_list)

        # %% [markdown]
        # ## Volatility

        # %%
        import ta.volatility as volatility

        # %% [markdown]
        # **Bollinger Bands**
        # 
        # https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands

        # %%
        for i in range(10):
            ds = volatility.bollinger_hband(self.log_pr_train[i], self.window_size, fillna=self.fillna)
            ds.name = 'bbh%d'%i
            features_list.append(ds)

        for i in range(10):
            ds = volatility.bollinger_hband_indicator(self.log_pr_train[i], self.window_size, fillna=self.fillna)
            ds.name = 'bbhi%d'%i
            features_list.append(ds)

        for i in range(10):
            ds = volatility.bollinger_lband(self.log_pr_train[i], self.window_size, fillna=self.fillna)
            ds.name = 'bbl%d'%i
            features_list.append(ds)

        for i in range(10):
            ds = volatility.bollinger_lband_indicator(self.log_pr_train[i], self.window_size, fillna=self.fillna)
            ds.name = 'bbli%d'%i
            features_list.append(ds)

        for i in range(10):
            ds = volatility.bollinger_mavg(self.log_pr_train[i], self.window_size, fillna=self.fillna)
            ds.name = 'bbm%d'%i
            features_list.append(ds)

        for i in range(10):
            ds = volatility.bollinger_pband(self.log_pr_train[i], self.window_size, fillna=self.fillna)
            ds.name = 'bbp%d'%i
            features_list.append(ds)

        for i in range(10):
            ds = volatility.bollinger_wband(self.log_pr_train[i], self.window_size, fillna=self.fillna)
            ds.name = 'bbw%d'%i
            features_list.append(ds)

        len(features_list)

        # %% [markdown]
        # **Ulcer Index**
        # 
        # https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ulcer_index

        # %%
        for i in range(10):
            ds = volatility.ulcer_index(self.log_pr_train[i], self.window_size, self.fillna)
            ds.name = 'ulcer%d'%i
            features_list.append(ds)

        len(features_list)

        # %% [markdown]
        # # Trend

        # %%
        import ta.trend as trend

        # %% [markdown]
        # **Aroon Indicator**
        # 
        # Identify when trends are likely to change direction.
        # 
        # Aroon Up = ((N - Days Since N-day High) / N) x 100 Aroon Down = ((N - Days Since N-day Low) / N) x 100 Aroon Indicator = Aroon Up - Aroon Down
        # 
        # https://www.investopedia.com/terms/a/aroon.asp

        # %%
        for i in range(10):
            ds = trend.aroon_down(self.log_pr_train[i], self.window_size, self.fillna)
            ds.name = 'arron_down%d'%i
            features_list.append(ds)

        for i in range(10):
            ds = trend.aroon_up(self.log_pr_train[i], self.window_size, self.fillna)
            ds.name = 'arron_up%d'%i
            features_list.append(ds)

        for i in range(10):
            ds = trend.AroonIndicator(self.log_pr_train[i], self.window_size, self.fillna).aroon_indicator()
            ds.name = 'arron_id%d'%i
            features_list.append(ds)

        len(features_list)

        # %% [markdown]
        # **Detrended Price Oscillator (DPO)**
        # 
        # Is an indicator designed to remove trend from price and make it easier to identify cycles.
        # 
        # http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci

        # %%
        for i in range(10):
            ds = trend.dpo(self.log_pr_train[i], self.window_size, self.fillna)
            ds.name = 'dpo%d'%i
            features_list.append(ds)

        # %%
        len(features_list)

        # %% [markdown]
        # **EMA - Exponential Moving Average**

        # %%
        for i in range(10):
            ds = trend.ema_indicator(self.log_pr_train[i], self.window_size, self.fillna)
            ds.name = 'ema%d'%i
            features_list.append(ds)

        len(features_list)

        # %% [markdown]
        # **Moving Average Convergence Divergence (MACD)**
        # 
        # Is a trend-following momentum indicator that shows the relationship between two moving averages of prices.
        # 
        # https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd

        # %%
        for i in range(10):
            ds = trend.macd(self.log_pr_train[i], window_fast=self.window_size, fillna=self.fillna)
            ds.name = 'macd%d'%i
            features_list.append(ds)

        len(features_list)

        # %% [markdown]
        # **SMA - Simple Moving Average**
        # 
        # Parameters
        # close (pandas.Series) – dataset ‘Close’ column.
        # 
        # window (int) – n period.
        # 
        # fillna (bool) – if self.fillna, fill nan values.

        # %%
        for i in range(10):
            ds = trend.sma_indicator(self.log_pr_train[i], self.window_size, self.fillna)
            ds.name = 'sma%d'%i
            features_list.append(ds)

        len(features_list)

        # %% [markdown]
        # **Trix (TRIX)**
        # 
        # Shows the percent rate of change of a triple exponentially smoothed moving average.
        # 
        # http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:trix

        # %%
        for i in range(10):
            ds = trend.trix(self.log_pr_train[i], self.window_size, self.fillna)
            ds.name = 'trix%d'%i
            features_list.append(ds)

        len(features_list)

        # %% [markdown]
        # **WMA - Weighted Moving Average**

        # %%
        for i in range(10):
            ds = trend.wma_indicator(self.log_pr_train[i], self.window_size, self.fillna)
            ds.name = 'wma%d'%i
            features_list.append(ds)

        len(features_list)

        # %% [markdown]
        # Combine the feature into pandas dataframe

        # %%
        self.features = pd.concat(features_list, axis=1)
        return self.features

    def save(self):
        self.features.to_pickle("./features.pkl")




