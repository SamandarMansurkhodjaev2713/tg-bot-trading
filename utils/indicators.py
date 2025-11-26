import numpy as np
import pandas as pd

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    up = high.diff()
    down = low.diff()*-1
    plus_dm = np.where((up>down) & (up>0), up, 0.0)
    minus_dm = np.where((down>up) & (down>0), down, 0.0)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(n).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(n).mean() / atr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan) * 100
    adx = dx.rolling(n).mean().fillna(0)
    return adx

def boll_bw(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.Series:
    close = df["close"].astype(float)
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std().fillna(0)
    upper = ma + k*sd
    lower = ma - k*sd
    bw = (upper - lower) / ma.replace(0, np.nan)
    return bw.fillna(0)
