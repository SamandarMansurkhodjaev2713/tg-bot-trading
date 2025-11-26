import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from joblib import load
from services.preprocess import _indicators
from services.regime import classify_regime
from services.news import load_calendar, is_blocked
from services.sentiment import load_sentiment, get_sentiment
from sklearn.calibration import CalibratedClassifierCV


def _to_market_list(df: pd.DataFrame) -> List[Dict]:
    out = []
    for _, r in df.iterrows():
        out.append({
            "timestamp": r.get("timestamp").isoformat() if pd.notnull(r.get("timestamp")) else None,
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
        })
    return out


def _pair_from_name(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    return base.upper()


def generate_signals(datasets: Dict[str, pd.DataFrame], scaler_path: str, models: Dict[str, object], config: Dict = None, per_instrument: Dict[str, Dict] = None) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    cfg = config or {}
    prob_thr = float(cfg.get("prob_thr", 0.6))
    exp_thr = float(cfg.get("exp_thr", 0.0002))
    tp_mult = float(cfg.get("tp_mult", 2.0))
    sl_mult = float(cfg.get("sl_mult", 1.0))
    step = int(cfg.get("step", 1))
    scaler = load(scaler_path)
    calendar = load_calendar()
    senti = load_sentiment()
    signals: List[Dict] = []
    market_data: Dict[str, List[Dict]] = {}
    best_map = {}
    try:
        import json
        from pathlib import Path
        p = Path('reports') / 'best_signals_by_instrument.json'
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                best_map = json.load(f)
    except Exception:
        best_map = {}
    for name, df in datasets.items():
        if not {"open", "high", "low", "close"}.issubset(df.columns):
            continue
        pair = _pair_from_name(name)
        is_1h = ('1h' in str(name).lower())
        market_data[pair] = _to_market_list(df)
        df2 = df.copy()
        df2 = _indicators(df2)
        regimes = classify_regime(df2)
        cols = [
            "open","high","low","close","volume","ret1","sma_10","sma_20","ema_10","ema_20","rsi_14","macd_line","macd_signal","macd_hist","stoch_k","stoch_d",
        ]
        for c in cols:
            if c not in df2.columns:
                df2[c] = 0.0
        X = df2[cols].dropna()
        if len(X) < 50:
            continue
        use_fallback = False
        try:
            Xs = scaler.transform(X.values)
        except Exception:
            Xs = X.values
            use_fallback = True
        clf1 = models.get("FinWorld")
        clf2 = models.get("FLAG-Trader")
        reg = models.get("FinBloom-7B")
        if not use_fallback and clf1 and clf2 and reg:
            y1 = clf1.model.predict(Xs) if hasattr(clf1, "model") else clf1.predict(Xs)
            y2 = clf2.model.predict(Xs) if hasattr(clf2, "model") else clf2.predict(Xs)
            y_bin = (df2["close"].shift(-1) > df2["close"]).astype(int).iloc[-len(Xs):].values
            try:
                n = len(Xs)
                k = max(int(n*0.8), 10)
                Xs_tr = Xs[:k]
                y_tr = y_bin[:k]
                cal1 = CalibratedClassifierCV(clf1.model, method='isotonic', cv=3)
                cal2 = CalibratedClassifierCV(clf2.model, method='isotonic', cv=3)
                cal1.fit(Xs_tr, y_tr)
                cal2.fit(Xs_tr, y_tr)
                p1_full = cal1.predict_proba(Xs)
                p2_full = cal2.predict_proba(Xs)
                p1 = p1_full[:,1] if p1_full.shape[1]>1 else p1_full[:,0]
                p2 = p2_full[:,1] if p2_full.shape[1]>1 else p2_full[:,0]
            except Exception:
                p1 = clf1.model.predict_proba(Xs)[:,1] if hasattr(clf1, "model") and hasattr(clf1.model, "predict_proba") else np.ones(len(y1))*0.6
                p2 = clf2.model.predict_proba(Xs)[:,1] if hasattr(clf2, "model") and hasattr(clf2.model, "predict_proba") else np.ones(len(y2))*0.6
            ret5 = reg.model.predict(Xs) if hasattr(reg, "model") else reg.predict(Xs)
        else:
            adx_series = df2.get("adx_14", pd.Series([20.0]*len(df2))).values
            macd_h = df2.get("macd_hist", pd.Series([0.0]*len(df2))).values
            y1 = np.where((df2.get("sma_10").values > df2.get("ema_20").values) & (macd_h > 0), 1, -1)
            y2 = np.where((df2.get("sma_20").values > df2.get("ema_10").values) & (macd_h > 0), 1, -1)
            p_base = 0.5 + 0.005*np.clip(adx_series - 20.0, 0, 50) + 0.02*np.tanh(macd_h)
            p1 = np.clip(p_base, 0.51, 0.99)
            p2 = np.clip(p_base * 0.98, 0.51, 0.99)
            ret5 = pd.Series(df2.get("close")).pct_change(5).fillna(0.0).values
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            (df["high"] - df["low"]),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().fillna(tr)
        for i in range(0, len(Xs)-1, step):
            agree = int(y1[i]) == int(y2[i])
            prob = float((p1[i] + p2[i]) / 2.0)
            ts_val = df["timestamp"].iloc[i] if "timestamp" in df.columns else None
            cfg_i = cfg
            if per_instrument:
                cfg_i = per_instrument.get(pair, cfg) or per_instrument.get(name, cfg) or cfg
            if not per_instrument and best_map:
                bm = best_map.get(name) or best_map.get(pair)
                if bm and isinstance(bm, dict):
                    cfg_i = {**cfg_i, **bm.get('cfg', {})}
            vol = float(df2.get("vol_hist_20", pd.Series([0]*len(df2))).iloc[i]) if "vol_hist_20" in df2.columns else 0.0
            news_window = int(cfg_i.get("news_window", 180))
            news_window_high = int(cfg_i.get("news_window_high", 240))
            vol_thr = float(cfg_i.get("vol_high_thr", 0.025))
            nw_use = news_window_high if vol > vol_thr else news_window
            if ts_val is not None and is_blocked(pd.to_datetime(ts_val, utc=True, errors="coerce"), pair, calendar, nw_use):
                continue
            direction = 'long' if int(y1[i]) == 1 else 'short'
            sent = get_sentiment(pair, pd.to_datetime(ts_val, utc=True, errors="coerce") if ts_val is not None else None, senti)
            sent_k = float(cfg_i.get("sentiment_coef", 0.15))
            sent_thr = float(cfg_i.get("sentiment_filter", 0.3))
            prob = max(min(prob * (1.0 + sent_k * sent), 0.999), 0.001)
            if abs(sent) >= sent_thr:
                if direction == 'long' and sent < 0:
                    continue
                if direction == 'short' and sent > 0:
                    continue
            if not agree:
                continue
            recent = df.iloc[i]
            price = float(recent["close"]) if "close" in recent else float(recent.get("Open", 0.0))
            a = float(atr.iloc[i])
            exp = float(ret5[i])
            tl = bool(df2.get("sma_10", pd.Series([0]*len(df2))).iloc[i] > df2.get("ema_20", pd.Series([0]*len(df2))).iloc[i] and df2.get("macd_hist", pd.Series([0]*len(df2))).iloc[i] > 0)
            ts = bool(df2.get("sma_10", pd.Series([0]*len(df2))).iloc[i] < df2.get("ema_20", pd.Series([0]*len(df2))).iloc[i] and df2.get("macd_hist", pd.Series([0]*len(df2))).iloc[i] < 0)
            bb_up = float(df2.get("bb_upper", pd.Series([0]*len(df2))).iloc[i])
            bb_lo = float(df2.get("bb_lower", pd.Series([0]*len(df2))).iloc[i])
            macd_line = float(df2.get("macd_line", pd.Series([0]*len(df2))).iloc[i])
            macd_sig = float(df2.get("macd_signal", pd.Series([0]*len(df2))).iloc[i])
            macd_hist_i = float(df2.get("macd_hist", pd.Series([0]*len(df2))).iloc[i])
            macd_hist_prev = float(df2.get("macd_hist", pd.Series([0]*len(df2))).iloc[i-1]) if i>0 else macd_hist_i
            adx_val = float(df2.get("adx_14", pd.Series([0]*len(df2))).iloc[i])
            adx_min = float(cfg_i.get("adx_min", 25.0))
            if is_1h:
                adx_min = max(adx_min, 30.0)
            regm = str(regimes.iloc[i]) if i < len(regimes) else "unknown"
            if regm != "trend":
                continue
            if direction == 'long' and (not tl or adx_val < adx_min):
                continue
            if direction == 'short' and (not ts or adx_val < adx_min):
                continue
            bb_confirm = bool(cfg_i.get("bb_macd_confirm", False))
            if bb_confirm:
                bb_ok_long = bool(price >= bb_up and macd_line > macd_sig and macd_hist_i > 0 and macd_hist_prev <= macd_hist_i)
                bb_ok_short = bool(price <= bb_lo and macd_line < macd_sig and macd_hist_i < 0 and macd_hist_prev >= macd_hist_i)
                if direction == 'long' and not bb_ok_long:
                    continue
                if direction == 'short' and not bb_ok_short:
                    continue
            r1 = float(df2.get("ret1", pd.Series([0]*len(df2))).iloc[i])
            if direction == 'long' and r1 <= 0:
                continue
            if direction == 'short' and r1 >= 0:
                continue
            tp_m = tp_mult
            sl_m = sl_mult
            prob_thr_i = prob_thr
            exp_thr_i = exp_thr
            prob_thr_i = float(cfg_i.get("prob_thr", prob_thr))
            exp_thr_i = float(cfg_i.get("exp_thr", exp_thr))
            if regm == "trend":
                prob_thr_i = max(prob_thr_i - 0.01, 0.5)
            elif regm == "chop":
                tp_m = max(tp_m - 0.5, 1.0)
                prob_thr_i = prob_thr_i + 0.02
            elif regm == "volatile":
                tp_m = tp_m + 0.5
                sl_m = sl_m + 0.2
                exp_thr_i = exp_thr_i + 0.0001
            if sent > 0.2:
                prob_thr_i = max(prob_thr_i - 0.02, 0.5)
            if sent < -0.2:
                prob_thr_i = prob_thr_i + 0.02
                exp_thr_i = exp_thr_i + 0.0001
            tp = price + (tp_m * a) if direction == 'long' else price - (tp_m * a)
            sl = price - (sl_m * a) if direction == 'long' else price + (sl_m * a)
            rr = abs(tp - price) / max(1e-9, abs(price - sl))
            rr_min = float(cfg_i.get("rr_min", 2.5))
            if regm == "chop":
                rr_min = max(rr_min, 2.0)
            elif regm == "volatile":
                rr_min = max(rr_min - 0.8, 1.2)
            if rr < rr_min:
                continue
            if direction == 'long' and prob < prob_thr_i:
                continue
            if direction == 'short' and prob < prob_thr_i:
                continue
            if direction == 'long' and exp <= exp_thr_i:
                continue
            if direction == 'short' and exp >= -exp_thr_i:
                continue
            signals.append({
                "timestamp": (ts_val.isoformat() if ts_val is not None else None) or "1970-01-01T00:00:00",
                "pair": pair,
                "direction": direction,
                "entry_price": price,
                "stop_loss": sl,
                "take_profit": tp,
                "confidence": round(prob * 100.0, 2),
                "market_regime": regm,
                "sentiment": round(sent, 4),
            })
    return signals, market_data
