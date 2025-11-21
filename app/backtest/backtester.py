import pandas as pd
import numpy as np
from datetime import datetime
from ..services.quotes import get_ohlc
from ..features.indicators import compute_features
from ..models.training import analyze_signal
from ..exness.config import COSTS

def _simulate(pair: str, tf: str, df: pd.DataFrame, feats: pd.DataFrame):
    c = COSTS.get(pair, {})
    spread = c.get("spread", 0)
    commission = c.get("commission", 0)
    slippage = c.get("slippage", 0)
    pnl = []
    equity = 1.0
    for i in range(50, len(df)-1):
        block_df = df.iloc[:i+1]
        block_feats = feats.iloc[:i+1]
        rec = analyze_signal(pair, tf, block_df, block_feats)
        action = rec["action"]
        size = rec["size"]
        px = float(block_df["close"].iloc[-1])
        px_next = float(df["close"].iloc[i+1])
        cost = spread + commission + slippage
        r = 0.0
        if action == "buy":
            r = (px_next - px - cost) / px
        elif action == "sell":
            r = (px - px_next - cost) / px
        equity *= (1 + r * min(size, 1))
        pnl.append(r)
    s = pd.Series(pnl)
    dd = (s.cumsum().cummax() - s.cumsum()).max()
    vol = s.std() * np.sqrt(252) if len(s) else 0
    sr = s.mean() / (s.std() + 1e-9) if len(s) else 0
    sortino = s.mean() / (s[s<0].std() + 1e-9) if len(s) else 0
    cagr = (equity ** (252/len(s)) - 1) if len(s) else 0
    calmar = cagr / (dd + 1e-9) if dd else 0
    return {"Sharpe": float(sr), "Sortino": float(sortino), "MaxDD": float(dd), "CAGR": float(cagr), "Calmar": float(calmar), "hit_rate": float((s>0).mean() if len(s) else 0), "turnover": float(len(s))}

def run_backtest(pair: str, tf: str, start: str, end: str):
    df = get_ohlc(pair, tf, 2000)
    feats = compute_features(df)
    rep = _simulate(pair, tf, df, feats)
    return rep