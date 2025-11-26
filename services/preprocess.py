import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

def _winsorize_returns(df):
    r = df["close"].pct_change().fillna(0)
    q1 = r.quantile(0.01)
    q99 = r.quantile(0.99)
    r = r.clip(q1, q99)
    df["ret1"] = r
    return df.dropna()

def _indicators(df):
    close = df["close"]
    df["sma_10"] = close.rolling(10).mean()
    df["sma_20"] = close.rolling(20).mean()
    df["ema_10"] = close.ewm(span=10, adjust=False).mean()
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=14).mean()
    roll_down = down.ewm(span=14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_line"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = macd - signal
    low = df["low"]
    high = df["high"]
    k = (close - low.rolling(14).min()) / (high.rolling(14).max() - low.rolling(14).min() + 1e-9)
    df["stoch_k"] = 100 * k
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std().fillna(0)
    df["bb_upper"] = ma20 + 2.0 * sd20
    df["bb_lower"] = ma20 - 2.0 * sd20
    high9 = high.rolling(9).max()
    low9 = low.rolling(9).min()
    high26 = high.rolling(26).max()
    low26 = low.rolling(26).min()
    df["ichimoku_tenkan"] = (high9 + low9) / 2.0
    df["ichimoku_kijun"] = (high26 + low26) / 2.0
    df["obv"] = (df["volume"].fillna(0) * (close.diff().fillna(0) > 0).astype(int) - df["volume"].fillna(0) * (close.diff().fillna(0) < 0).astype(int)).cumsum()
    df["rv_20"] = close.pct_change().rolling(20).std().fillna(0)
    df["vol_hist_20"] = close.pct_change().rolling(20).std().fillna(0) * np.sqrt(252)
    dm_pos = (high.diff().clip(lower=0)).fillna(0)
    dm_neg = (-low.diff().clip(upper=0)).fillna(0)
    tr14 = tr.rolling(14).sum().replace(0, np.nan)
    di_pos = (100 * dm_pos.rolling(14).sum() / tr14).fillna(0)
    di_neg = (100 * dm_neg.rolling(14).sum() / tr14).fillna(0)
    dx = (100 * (di_pos - di_neg).abs() / (di_pos + di_neg + 1e-9)).fillna(0)
    df["adx_14"] = dx.rolling(14).mean().fillna(0)
    return df.dropna()

def build_dataset(datasets):
    frames = []
    for name, df in datasets.items():
        if not {"open", "high", "low", "close"}.issubset(df.columns):
            continue
        df = df.copy()
        df = _winsorize_returns(df)
        df = _indicators(df)
        df["future_return_5"] = df["close"].shift(-5) / df["close"] - 1.0
        df["cls_target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df["source"] = name
        frames.append(df.dropna())
    data = pd.concat(frames, axis=0, ignore_index=True)
    cols_keep = [
        "open","high","low","close","volume","ret1","sma_10","sma_20","ema_10","ema_20","rsi_14","macd_line","macd_signal","macd_hist","stoch_k","stoch_d","future_return_5","cls_target",
        "atr_14","bb_upper","bb_lower","ichimoku_tenkan","ichimoku_kijun","obv","rv_20"
    ]
    data = data[cols_keep].dropna()
    return data

def preprocess(data):
    X = data.drop(["future_return_5", "cls_target"], axis=1)
    y_cls = data["cls_target"].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    dump(scaler, "models/scaler_main.pkl")
    X_train, X_val, y_train, y_val = train_test_split(Xs, y_cls, test_size=0.2, shuffle=False)
    return X_train, X_val, y_train, y_val
