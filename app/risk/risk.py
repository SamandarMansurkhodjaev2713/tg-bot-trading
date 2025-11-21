import numpy as np
from ..exness.config import COSTS

def position_size(pair: str, tf: str, feats):
    atr = float(feats["atr_14"].iloc[-1])
    c = COSTS.get(pair, {})
    leverage = c.get("leverage", 100)
    risk = 0.01
    if atr == 0:
        return 0.0
    vol_filter = float(feats["vol_hist_20"].iloc[-1])
    if vol_filter > 0.03:
        risk = 0.005
    size = risk * leverage / max(atr, 1e-6)
    return float(min(size, c.get("max_size", 10)))

def allow_trade(pair: str, feats):
    c = COSTS.get(pair, {})
    spread = c.get("spread", 0.0002)
    atr = float(feats["atr_14"].iloc[-1])
    if spread > atr * 0.3:
        return False
    return True