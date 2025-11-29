import pickle
import os
import numpy as np
import pandas as pd
import asyncio
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, List
from ..features.indicators import compute_features
from tools.data_sources import DataManager
from ..services.quotes import get_ohlc
from ..utils.env import load_env

_models: Dict[str, Any] = {}

def _label(df: pd.DataFrame) -> pd.Series:
    r = df["close"].pct_change().shift(-1)
    y = np.where(r > 0.0002, 1, np.where(r < -0.0002, -1, 0))
    return pd.Series(y, index=df.index)

def _to_Xy(feats: pd.DataFrame) -> tuple:
    y = _label(feats)
    X = feats.drop(columns=[c for c in ["open","high","low","close","adj_close","volume"] if c in feats.columns])
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    return X, y

async def _get_df_multi(pair: str, tf: str, limit: int) -> pd.DataFrame:
    load_env()
    cfg = {
        'alphavantage_api_key': os.getenv('ALPHAVANTAGE_API_KEY', ''),
        'histdata_path': os.getenv('HISTDATA_PATH', 'data/histdata'),
        'truefx_path': os.getenv('TRUEFX_PATH', 'data/truefx')
    }
    try:
        from ..services import mt5 as mt5svc
        df_mt5 = mt5svc.get_df(pair, tf, limit)
        if isinstance(df_mt5, pd.DataFrame) and len(df_mt5) > 0:
            return df_mt5
    except Exception:
        pass
    dm = DataManager(cfg)
    data = await dm.get_merged_market_data(pair, tf, limit)
    if not data:
        alt = pair.replace('/', '')
        data = await dm.get_merged_market_data(alt, tf, limit)
    if not data:
        df = get_ohlc(pair, tf, limit)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    df = pd.DataFrame(data)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.dropna(subset=['timestamp']).sort_values('timestamp').set_index('timestamp')
    for c in ['open','high','low','close','volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna()
    return df

def _synthetic_df(limit: int, tf: str) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=limit, freq='15min' if tf=='15m' else '1h')
    base = 100.0
    rnd = np.random.normal(0, 0.1, size=len(idx)).cumsum()
    close = base + rnd
    openp = close * (1 - np.random.normal(0, 0.001, size=len(idx)))
    high = np.maximum(openp, close) * (1 + np.random.uniform(0, 0.002, size=len(idx)))
    low = np.minimum(openp, close) * (1 - np.random.uniform(0, 0.002, size=len(idx)))
    vol = np.random.randint(500, 1500, size=len(idx))
    return pd.DataFrame({"open": openp, "high": high, "low": low, "close": close, "volume": vol}, index=idx)

async def train_pairs(pairs: List[str], tf: str) -> Dict[str, Any]:
    res = []
    for p in pairs:
        df = await _get_df_multi(p, tf, 2000)
        if df is None or len(df) == 0:
            df = _synthetic_df(600, tf)
        if df is None or len(df) < 120:
            res.append({"pair": p, "trained": False, "samples": 0})
            continue
        feats = compute_features(df)
        X, y = _to_Xy(feats)
        if len(X) < 200:
            res.append({"pair": p, "trained": False, "samples": int(len(X))})
            continue
        clf = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=200, random_state=42))])
        clf.fit(X.iloc[:-1], y.iloc[:-1])
        _models[p] = {"clf": clf, "tf": tf, "features": list(X.columns)}
        res.append({"pair": p, "trained": True, "samples": int(len(X))})
    try:
        pickle.dump(_models, open("ai_models.pkl", "wb"))
    except Exception:
        pass
    return {"results": res}

async def predict_pair(pair: str, tf: str) -> Dict[str, Any]:
    if pair not in _models or _models.get(pair, {}).get("tf") != tf:
        await train_pairs([pair], tf)
    m = _models.get(pair)
    df = await _get_df_multi(pair, tf, 600)
    if df is None or len(df) == 0:
        df = _synthetic_df(300, tf)
    feats = compute_features(df)
    X, y = _to_Xy(feats)
    if len(X) == 0:
        return {"pair": pair, "tf": tf, "error": "no_data"}
    last = X.iloc[-1:]
    clf = m["clf"] if m else Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=200, random_state=42))]).fit(X, y)
    pred = int(clf.predict(last)[0])
    try:
        proba = float(max(clf.predict_proba(last)[0]))
    except Exception:
        proba = 0.5
    atr = float(feats["atr_14"].iloc[-1]) if "atr_14" in feats.columns else float(np.std(df["close"].pct_change().tail(14)))
    px = float(df["close"].iloc[-1])
    sl = px - atr * 1.5 if pred == 1 else px + atr * 1.5
    tp = px + atr * 2.0 if pred == 1 else px - atr * 2.0
    vol = float(np.std(df["close"].pct_change().dropna().tail(50))) if len(df) > 50 else 0.0
    macd = float(feats.get("macd", pd.Series([0])).iloc[-1])
    macd_sig = float(feats.get("macd_signal", pd.Series([0])).iloc[-1])
    macd_bull = macd >= macd_sig
    return {
        "pair": pair,
        "tf": tf,
        "action": "buy" if pred==1 else "sell" if pred==-1 else "hold",
        "sl": sl,
        "tp": tp,
        "rsi": float(feats.get("rsi_14", pd.Series([50])).iloc[-1]),
        "adx": float(feats.get("adx_14", pd.Series([20])).iloc[-1]),
        "price": px,
        "vol": vol,
        "macd_bull": macd_bull,
        "probability": proba
    }
