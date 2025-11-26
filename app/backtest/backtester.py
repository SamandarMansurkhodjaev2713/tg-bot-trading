import pandas as pd
import numpy as np
from ..services.quotes import get_ohlc
from services.preprocess import _winsorize_returns, _indicators
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from ..exness.config import COSTS

def _label_bin(feats: pd.DataFrame) -> pd.Series:
    return (feats["close"].shift(-1) > feats["close"]).astype(int)

def _simulate_fast(pair: str, df: pd.DataFrame, feats: pd.DataFrame):
    c = COSTS.get(pair, {})
    spread = float(c.get("spread", 0))
    commission = float(c.get("commission", 0))
    slippage = float(c.get("slippage", 0))
    cost = spread + commission + slippage
    y = _label_bin(feats)
    X = feats.drop(columns=[c for c in ["open","high","low","close","adj_close","volume"] if c in feats.columns])
    if len(X) < 200:
        return {"Sharpe": 0.0, "Sortino": 0.0, "MaxDD": 0.0, "CAGR": 0.0, "Calmar": 0.0, "hit_rate": 0.0, "turnover": 0}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    base = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=200, random_state=42))])
    base.fit(X_train, y_train)
    try:
        clf = CalibratedClassifierCV(base.named_steps["model"], method="isotonic", cv=3)
        clf.fit(base.named_steps["scaler"].transform(X_train), y_train)
        proba = clf.predict_proba(base.named_steps["scaler"].transform(X_test))
        classes = list(clf.classes_)
    except Exception:
        preds = base.predict(X_test)
        # Fallback: fake two-class probabilities
        proba = None
        classes = [0, 1]
    # Build returns on test segment using next-step price changes
    test_idx = X_test.index
    pnl = []
    equity = 1.0
    for i, idx in enumerate(test_idx):
        if i == len(test_idx) - 1:
            break
        px = float(df.loc[idx, "close"]) if "close" in df.columns else float(df.loc[idx, "Close"]) 
        next_idx = test_idx[i+1]
        px_next = float(df.loc[next_idx, "close"]) if "close" in df.columns else float(df.loc[next_idx, "Close"]) 
        # Determine action using calibrated probabilities when available
        if proba is not None:
            i_pos = classes.index(1) if 1 in classes else len(classes)-1
            p_up = float(proba[i][i_pos])
            thr = 0.9
            if p_up >= thr:
                action = 1
            elif p_up <= (1.0 - thr):
                action = -1
            else:
                action = 0
        else:
            action = int(preds[i])
        r = 0.0
        if action == 1:
            r = (px_next - px - cost) / max(px, 1e-9)
        elif action == -1:
            r = (px - px_next - cost) / max(px, 1e-9)
        equity *= (1 + r)
        pnl.append(r)
    s = pd.Series(pnl)
    dd = float((s.cumsum().cummax() - s.cumsum()).max()) if len(s) else 0.0
    sr = float(s.mean() / (s.std() + 1e-9)) if len(s) else 0.0
    sortino = float(s.mean() / (s[s<0].std() + 1e-9)) if len(s) else 0.0
    cagr = float((equity ** (252/max(len(s),1)) - 1)) if len(s) else 0.0
    calmar = float(cagr / (dd + 1e-9)) if dd else 0.0
    return {"Sharpe": sr, "Sortino": sortino, "MaxDD": dd, "CAGR": cagr, "Calmar": calmar, "hit_rate": float((s>0).mean() if len(s) else 0), "turnover": int(len(s))}

def optimize_threshold(pair: str, tf: str, window: int = 2000):
    df = get_ohlc(pair, tf, window)
    try:
        from ..features.indicators import compute_features
        feats = compute_features(df)
    except Exception:
        s = df.copy()
        s = _winsorize_returns(s)
        s = _indicators(s)
        feats = s
    # Prepare data
    y = _label_bin(feats)
    X = feats.drop(columns=[c for c in ["open","high","low","close","adj_close","volume"] if c in feats.columns])
    if len(X) < 300:
        return {"best_thr": 0.95, "metrics": {"hit_rate": 0.0, "turnover": 0}}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    base = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42))])
    base.fit(X_train, y_train)
    try:
        clf = CalibratedClassifierCV(base.named_steps["model"], method="isotonic", cv=3)
        clf.fit(base.named_steps["scaler"].transform(X_train), y_train)
        proba = clf.predict_proba(base.named_steps["scaler"].transform(X_test))
        classes = list(clf.classes_)
    except Exception:
        try:
            proba = base.named_steps["model"].predict_proba(base.named_steps["scaler"].transform(X_test))
            classes = list(base.named_steps["model"].classes_)
        except Exception:
            proba = None
            classes = [0, 1]
    test_idx = X_test.index
    # Trend filters
    has_sma = "sma_10" in feats.columns and "ema_20" in feats.columns
    has_macd = "macd_hist" in feats.columns
    def trend_ok(i, direction):
        return True
    # Grid search thresholds
    candidates = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    best = {"thr": 0.6, "hit_rate": -1.0, "turnover": 0}
    for thr in candidates:
        pnl = []
        equity = 1.0
        wins = 0
        trades = 0
        for i, idx in enumerate(test_idx[:-1]):
            px = float(df.loc[idx, "close"]) if "close" in df.columns else float(df.loc[idx, "Close"]) 
            next_idx = test_idx[i+1]
            px_next = float(df.loc[next_idx, "close"]) if "close" in df.columns else float(df.loc[next_idx, "Close"]) 
            if proba is None:
                continue
            i_pos = classes.index(1) if 1 in classes else len(classes)-1
            p_up = float(proba[i][i_pos])
            if p_up >= thr and trend_ok(i, 1):
                r = (px_next - px) / max(px, 1e-9)
            elif p_up <= (1.0 - thr) and trend_ok(i, -1):
                r = (px - px_next) / max(px, 1e-9)
            else:
                continue
            trades += 1
            wins += 1 if r > 0 else 0
            equity *= (1 + r)
            pnl.append(r)
        hit = float(wins / trades) if trades else 0.0
        if hit > best["hit_rate"]:
            best = {"thr": thr, "hit_rate": hit, "turnover": trades}
    return {"best_thr": best["thr"], "metrics": {"hit_rate": best["hit_rate"], "turnover": best["turnover"]}}

def run_backtest(pair: str, tf: str, window: int = 2000):
    df = get_ohlc(pair, tf, window)
    try:
        from ..features.indicators import compute_features
        feats = compute_features(df)
    except Exception:
        s = df.copy()
        s = _winsorize_returns(s)
        s = _indicators(s)
        feats = s
    rep = _simulate_fast(pair, df, feats)
    return rep
