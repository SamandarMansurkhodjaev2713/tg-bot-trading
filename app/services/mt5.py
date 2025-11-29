import os
import pandas as pd

def _get_env(k: str) -> str:
    v = os.getenv(k, "")
    return v.strip() if isinstance(v, str) else ""

def _map_tf(tf: str):
    try:
        import MetaTrader5 as mt5
    except Exception:
        return None
    m = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }
    return m.get(tf)

def init() -> bool:
    try:
        import MetaTrader5 as mt5
    except Exception:
        return False
    path = _get_env("MT5_PATH")
    login = _get_env("MT5_LOGIN")
    password = _get_env("MT5_PASSWORD")
    server = _get_env("MT5_SERVER")
    if not path:
        return False
    try:
        lval = int(login) if login else 0
    except Exception:
        lval = 0
    ok = mt5.initialize(path=path, login=lval, password=password, server=server)
    return bool(ok)

def get_df(pair: str, tf: str, limit: int) -> pd.DataFrame:
    try:
        import MetaTrader5 as mt5
    except Exception:
        return pd.DataFrame()
    if not init():
        return pd.DataFrame()
    symbol = pair.replace("/", "")
    mt5.symbol_select(symbol, True)
    tfc = _map_tf(tf)
    if tfc is None:
        return pd.DataFrame()
    try:
        from datetime import datetime
        rates = mt5.copy_rates_from_pos(symbol, tfc, 0, limit)
    except Exception:
        rates = None
    if not rates:
        return pd.DataFrame()
    import numpy as np
    arr = rates if isinstance(rates, list) else list(rates)
    if not arr:
        return pd.DataFrame()
    t = [r[0] if not hasattr(r, "time") else r.time for r in arr]
    o = [r[1] if not hasattr(r, "open") else r.open for r in arr]
    h = [r[2] if not hasattr(r, "high") else r.high for r in arr]
    l = [r[3] if not hasattr(r, "low") else r.low for r in arr]
    c = [r[4] if not hasattr(r, "close") else r.close for r in arr]
    v = [r[5] if not hasattr(r, "tick_volume") else r.tick_volume for r in arr]
    idx = pd.to_datetime(pd.Series(t), unit="s", utc=True)
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)
    df = df.sort_index()
    return df
