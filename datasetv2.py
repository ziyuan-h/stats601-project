{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "33099c68",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "log_pr_file = './log_price.df'\n",
    "volu_usd_file = './volume_usd.df'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "b233d4a6",
   "metadata": {},
   "outputs": [],
   "source": [
    "log_pr = pd.read_pickle(log_pr_file)\n",
    "volu = pd.read_pickle(volu_usd_file)\n",
    "log_pr.columns = ['log_pr_%d'%i for i in range(10)]\n",
    "volu.columns = ['volu_%d'%i for i in range(10)]\n",
    "\n",
    "data = pd.concat([log_pr, volu], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "a5fe0d39",
   "metadata": {},
   "outputs": [],
   "source": [
    "def split_data(data:pd.DataFrame, test_pct:float):\n",
    "    assert test_pct > 0 and test_pct < 1\n",
    "    test_size = len(data) * test_pct\n",
    "    return data[:test_size], data[test_size:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "8eceeb4e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def formulize_data(data:pd.DataFrame, window_size=1440, step=10) -> np.array:\n",
    "    N = len(data)\n",
    "    train_index = np.arange(0, window_size)[np.newaxis, :] + step * np.arange(0, (N - window_size - 30) // step)[:, np.newaxis]\n",
    "    return_index = train_index[:, -1] + 30\n",
    "    return data.values[train_index], data.values[return_index]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "581f6752",
   "metadata": {},
   "outputs": [],
   "source": [
    "def rate_of_change(data:pd.DataFrame, periods):\n",
    "    return data.pct_change(periods)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "a7191f3e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def moving_average(data:pd.DataFrame, window_size):\n",
    "    return data.rolling(window_size).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "d0eaa55a",
   "metadata": {},
   "outputs": [],
   "source": [
    "def z_score(data:pd.DataFrame, window_size):\n",
    "    assert window_size > 1\n",
    "    return (data - data.rolling(window=window_size).mean()) / \\\n",
    "            data.rolling(window=window_size).std()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "2ae44775",
   "metadata": {},
   "outputs": [],
   "source": [
    "def moving_sum(data:pd.DataFrame, window_size):\n",
    "    return data.rolling(window_size).sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "bc8740b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "def sign(data:pd.DataFrame):\n",
    "    return np.sign(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "5129576e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def binning(data:pd.DataFrame, n_bins):\n",
    "    bin_fn = lambda y: pd.qcut(y, q=n_bins, labels=range(1, n_bins+1))\n",
    "    return data.apply(bin_fn)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3f60cc2f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
