import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from joblib import dump
from pathlib import Path
import json
from ..risk.risk import position_size
from ..exness.config import COSTS
from ..services.quotes import get_ohlc
from services.preprocess import _winsorize_returns, _indicators

def _label(df: pd.DataFrame):
    r = df["close"].pct_change().shift(-1)
    y = np.where(r > 0.0002, 1, np.where(r < -0.0002, -1, 0))
    return pd.Series(y, index=df.index)

def analyze_signal(pair: str, tf: str, df: pd.DataFrame, feats: pd.DataFrame):
    y = _label(feats)
    X = feats.drop(columns=[c for c in ["open","high","low","close","adj_close","volume"] if c in feats.columns])
    tscv = TimeSeriesSplit(n_splits=3)
    clf = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
    best = None
    for train_idx, test_idx in tscv.split(X):
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        p = clf.predict(X.iloc[test_idx])
        a = accuracy_score(y.iloc[test_idx], p)
        best = clf
    last = X.iloc[-1:]
    pred = int(best.predict(last)[0])
    try:
        proba = float(max(best.predict_proba(last)[0]))
    except Exception:
        proba = 0.5
    size = position_size(pair, tf, feats)
    price = float(df["close"].iloc[-1])
    atr = float(feats["atr_14"].iloc[-1]) if "atr_14" in feats.columns else float(np.std(df["close"].pct_change().tail(14)))
    sl = float(price - atr * 1.5) if pred == 1 else float(price + atr * 1.5)
    tp = float(price + atr * 2.0) if pred == 1 else float(price - atr * 2.0)
    macd_bull = False
    try:
        macd_bull = bool(feats["macd"].iloc[-1] >= feats["macd_signal"].iloc[-1])
    except Exception:
        macd_bull = False
    try:
        bb_pos = float(feats["bb_perc"].iloc[-1] * 100)
    except Exception:
        bb_pos = 0.0
    return {
        "pair": pair,
        "tf": tf,
        "action": "buy" if pred==1 else "sell" if pred==-1 else "hold",
        "size": size,
        "sl": sl,
        "tp": tp,
        "price": price,
        "atr": atr,
        "macd_bull": macd_bull,
        "bb_pos": bb_pos,
        "probability": proba,
        "explanation": {"indicators": {"rsi": float(feats["rsi_14"].iloc[-1]), "adx": float(feats["adx_14"].iloc[-1])}}
    }

def train_models(pairs: list[str], tf: str):
    reports = []
    for p in pairs:
        df = get_ohlc(p, tf, 2000)
        try:
            from ..features.indicators import compute_features
            feats = compute_features(df)
        except Exception:
            s = df.copy()
            s = _winsorize_returns(s)
            s = _indicators(s)
            feats = s
        y = _label(feats)
        X = feats.drop(columns=[c for c in ["open","high","low","close","adj_close","volume"] if c in feats.columns])
        tscv = TimeSeriesSplit(n_splits=3)
        clf = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=200, random_state=42))])
        scores = []
        for tr, te in tscv.split(X):
            clf.fit(X.iloc[tr], y.iloc[tr])
            pred = clf.predict(X.iloc[te])
            scores.append(accuracy_score(y.iloc[te], pred))
        reports.append({"pair": p, "accuracy": float(np.mean(scores))})
    return {"results": reports}

def train_and_save(pair: str, tf: str, window: int, out_dir: Path) -> dict:
    df = get_ohlc(pair, tf, window)
    try:
        from ..features.indicators import compute_features
        feats = compute_features(df)
    except Exception:
        s = df.copy()
        s = _winsorize_returns(s)
        s = _indicators(s)
        feats = s
    y = _label(feats)
    X = feats.drop(columns=[c for c in ["open","high","low","close","adj_close","volume"] if c in feats.columns])
    tscv = TimeSeriesSplit(n_splits=3)
    grid = ParameterGrid({"n_estimators": [100, 200, 300], "max_depth": [None, 5, 8]})
    best_score = -1.0
    best_params = None
    best_clf = None
    for params in grid:
        clf = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(random_state=42, **params))])
        scores = []
        for tr, te in tscv.split(X):
            clf.fit(X.iloc[tr], y.iloc[tr])
            pred = clf.predict(X.iloc[te])
            scores.append(accuracy_score(y.iloc[te], pred))
        m = float(np.mean(scores))
        if m > best_score:
            best_score = m
            best_params = params
            best_clf = clf
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{pair.replace('/', '')}_{tf}_rf.pkl"
    dump(best_clf, model_path.as_posix())
    metrics = {"pair": pair, "tf": tf, "accuracy": best_score, "params": best_params}
    with open(out_dir / f"{pair.replace('/', '')}_{tf}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f)
    return metrics

def main():
    root = Path('.')
    models_dir = root / 'models'
    reports_dir = root / 'reports'
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    pairs = ["EUR/USD", "XAU/USD"]
    tfs = ["15m", "1h"]
    def _iterative(cycles: int = 6):
        from app.backtest.backtester import run_backtest
        logs = []
        base_window = 600
        for c in range(cycles):
            cycle_res = {"cycle": c+1, "pairs": []}
            for p in pairs:
                for tf in tfs:
                    w = base_window + c * 200
                    bt = run_backtest(p, tf, window=w)
                    tr = train_and_save(p, tf, w, models_dir)
                    cycle_res["pairs"].append({"pair": p, "tf": tf, "backtest": bt, "train": tr})
            logs.append(cycle_res)
        with open(reports_dir / 'iterative_training_results.json', 'w', encoding='utf-8') as f:
            json.dump(logs, f)
    _iterative(6)

if __name__ == "__main__":
    main()
