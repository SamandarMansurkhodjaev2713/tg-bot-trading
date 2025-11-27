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
from ..features.indicators import compute_features
from ..services.news import news_summary
from tools.data_sources import DataManager
import asyncio
from ..features.indicators import compute_features
from ..services.news import news_summary
from tools.data_sources import DataManager
import asyncio

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

async def unified_analysis(pair: str, tf: str, window: int = 500):
    try:
        df = get_ohlc(pair, tf, window)
    except Exception:
        df = None
    if df is None or len(df) < 50:
        dm = DataManager({})
        data = await dm.get_merged_market_data(pair, tf, window)
        import pandas as pd
        if data:
            d = pd.DataFrame(data)
            d['timestamp'] = pd.to_datetime(d['timestamp'])
            d = d.sort_values('timestamp').set_index('timestamp')
            df = d
        if df is None or len(df) < 50:
            alt = pair.replace('/', '')
            data2 = await dm.get_merged_market_data(alt, tf, window)
            if data2:
                d2 = pd.DataFrame(data2)
                d2['timestamp'] = pd.to_datetime(d2['timestamp'])
                d2 = d2.dropna(subset=['timestamp']).sort_values('timestamp').set_index('timestamp')
                df = d2
        if df is None or len(df) < 50:
            alt = pair.replace('/', '')
            data = await dm.get_merged_market_data(alt, tf, window)
            if data:
                d = pd.DataFrame(data)
                d['timestamp'] = pd.to_datetime(d['timestamp'])
                d = d.dropna(subset=['timestamp']).sort_values('timestamp').set_index('timestamp')
                df = d
    if df is None or len(df) < 10:
        nw = news_summary('XAU/USD', 24)
        return {
            'pair': pair,
            'tf': tf,
            'news': { 'summary': 'недостаточно рыночных данных', 'influence': 'n/a', 'sources': nw.get('rss', [])[:5] },
            'history': { 'trend': 'n/a', 'volatility_atr': 0.0, 'levels': { 'piv_low': None, 'piv_high': None } },
            'sentiment': { 'risk_mode': 'n/a', 'sp500_change': 0.0 },
            'fear_greed': { 'source': 'CNN (не подключено)', 'value': None },
            'institutional': { 'cot': None, 'flows': None },
            'indicators': {},
            'final': { 'direction': 'none', 'confidence': 0.0, 'sl': None, 'sl_reason': 'нет данных', 'tp': None, 'tp_reason': 'нет данных', 'price': None }
        }
    feats = compute_features(df)
    ema20 = float(feats['ema_20'].iloc[-1]) if 'ema_20' in feats.columns else float(df['close'].iloc[-1])
    ema50 = float(feats['ema_50'].iloc[-1]) if 'ema_50' in feats.columns else ema20
    rsi = float(feats['rsi_14'].iloc[-1]) if 'rsi_14' in feats.columns else 50.0
    adx = float(feats['adx_14'].iloc[-1]) if 'adx_14' in feats.columns else 20.0
    di_pos = float(feats['di_pos'].iloc[-1]) if 'di_pos' in feats.columns else 20.0
    di_neg = float(feats['di_neg'].iloc[-1]) if 'di_neg' in feats.columns else 20.0
    bb_h = float(feats['bb_h'].iloc[-1]) if 'bb_h' in feats.columns else float(df['close'].iloc[-1])
    bb_l = float(feats['bb_l'].iloc[-1]) if 'bb_l' in feats.columns else float(df['close'].iloc[-1])
    px = float(df['close'].iloc[-1])
    atr = float(feats['atr_14'].iloc[-1]) if 'atr_14' in feats.columns else float(df['close'].pct_change().rolling(14).std().iloc[-1] * px)
    macd = float(feats['macd'].iloc[-1]) if 'macd' in feats.columns else 0.0
    macd_signal = float(feats['macd_signal'].iloc[-1]) if 'macd_signal' in feats.columns else 0.0
    macd_bull = macd >= macd_signal
    # Index sentiment via S&P500 (risk-on/off proxy)
    try:
        sp_dm = DataManager({})
        sp = await sp_dm.get_market_data('US500', '1d', 3)
        sp_ch = 0.0
        if sp and len(sp) >= 2:
            import pandas as pd
            sp_df = pd.DataFrame(sp)
            sp_ch = (float(sp_df['close'].iloc[-1]) - float(sp_df['close'].iloc[-2])) / float(sp_df['close'].iloc[-2])
    except Exception:
        sp_ch = 0.0
    # DXY change (risk proxy via USD strength)
    try:
        dxy_dm = DataManager({})
        dxy = await dxy_dm.get_market_data('DXY', '1d', 3)
        dxy_ch = 0.0
        if dxy and len(dxy) >= 2:
            import pandas as pd
            dxy_df = pd.DataFrame(dxy)
            dxy_ch = (float(dxy_df['close'].iloc[-1]) - float(dxy_df['close'].iloc[-2])) / float(dxy_df['close'].iloc[-2])
    except Exception:
        dxy_ch = 0.0
    # VIX-based Fear & Greed proxy
    try:
        vix_dm = DataManager({})
        vix = await vix_dm.get_market_data('VIX', '1d', 60)
        fg_val = None
        if vix and len(vix) >= 10:
            import pandas as pd
            vix_df = pd.DataFrame(vix)
            vals = pd.to_numeric(vix_df['close'], errors='coerce').dropna()
            if len(vals) >= 10:
                last = float(vals.iloc[-1])
                rank = float((vals <= last).mean())  # percentile
                greed = int(max(0, min(100, round((1.0 - rank) * 100))))
                fg_val = greed
    except Exception:
        fg_val = None
    risk_off = (sp_ch < 0) or (dxy_ch > 0)
    buy_score = 0.0
    sell_score = 0.0
    if ema20 > ema50:
        buy_score += 0.25
    elif ema20 < ema50:
        sell_score += 0.25
    if macd_bull:
        buy_score += 0.2
    else:
        sell_score += 0.2
    if rsi <= 30:
        buy_score += 0.15
    elif rsi >= 70:
        sell_score += 0.15
    if adx >= 25 and di_pos > di_neg:
        buy_score += 0.2
    elif adx >= 25 and di_neg > di_pos:
        sell_score += 0.2
    if px <= bb_l:
        buy_score += 0.1
    elif px >= bb_h:
        sell_score += 0.1
    if risk_off:
        buy_score += 0.1
    else:
        sell_score += 0.1
    total = max(buy_score, sell_score)
    direction = 'buy' if buy_score > sell_score else 'sell'
    confident = total >= 0.92
    piv_low = float(feats['piv_low'].iloc[-1]) if 'piv_low' in feats.columns else float(df['low'].rolling(5).min().iloc[-1])
    piv_high = float(feats['piv_high'].iloc[-1]) if 'piv_high' in feats.columns else float(df['high'].rolling(5).max().iloc[-1])
    if direction == 'buy':
        sl = piv_low - atr * 0.3
        tp = piv_high if piv_high > px else px + atr * 2.0
        sl_reason = 'ниже ближайшей поддержки (свинг‑лоу) с запасом по ATR'
        tp_reason = 'к ближайшему сопротивлению (свинг‑хай) или ATR×2'
    else:
        sl = piv_high + atr * 0.3
        tp = piv_low if piv_low < px else px - atr * 2.0
        sl_reason = 'выше ближайшего сопротивления (свинг‑хай) с запасом по ATR'
        tp_reason = 'к ближайшей поддержке (свинг‑лоу) или ATR×2'
    nw = news_summary('XAU/USD', 24)
    news_items = nw.get('rss', [])
    news_brief = 'реальные важные новости: ' + str(len(news_items)) if news_items else 'новостной фон спокоен'
    influence = 'умеренно положительное' if risk_off else 'умеренно отрицательное'
    # COT institutional data (latest)
    try:
        cot_dm = DataManager({})
        cot = await cot_dm.get_cot_gold()
    except Exception:
        cot = {}
    structured = {
        'pair': pair,
        'tf': tf,
        'news': { 'summary': news_brief, 'influence': influence, 'sources': news_items[:5] },
        'history': { 'trend': 'EMA20>EMA50' if ema20>ema50 else 'EMA20<EMA50', 'volatility_atr': atr, 'levels': { 'piv_low': piv_low, 'piv_high': piv_high } },
        'sentiment': { 'risk_mode': 'risk-off' if risk_off else 'risk-on', 'sp500_change': sp_ch, 'dxy_change': dxy_ch },
        'fear_greed': { 'source': 'VIX proxy', 'value': fg_val },
        'institutional': { 'cot': cot, 'flows': None },
        'indicators': { 'ema20': ema20, 'ema50': ema50, 'rsi': rsi, 'adx': adx, 'macd_bull': macd_bull, 'bb_low_touch': px<=bb_l, 'bb_high_touch': px>=bb_h },
        'final': { 'direction': direction if confident else 'none', 'confidence': total, 'sl': sl, 'sl_reason': sl_reason, 'tp': tp, 'tp_reason': tp_reason, 'price': px }
    }
    return structured

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
async def backtest_unified(pair: str, tf: str, window: int = 2000) -> dict:
    try:
        dm = DataManager({})
        data = await dm.get_merged_market_data(pair, tf, window)
        import pandas as pd
        if not data or len(data) < 200:
            return { 'pair': pair, 'tf': tf, 'trades': 0, 'win_rate': 0.0, 'total_return': 0.0 }
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.dropna(subset=['open','high','low','close']).sort_values('timestamp').set_index('timestamp')
        feats = compute_features(df)
        wins = 0
        losses = 0
        ret_sum = 0.0
        trades = 0
        n = len(df)
        for i in range(60, n-1):
            row = feats.iloc[i]
            px = float(df['close'].iloc[i])
            ema20 = float(row.get('ema_20', px))
            ema50 = float(row.get('ema_50', ema20))
            rsi = float(row.get('rsi_14', 50.0))
            adx = float(row.get('adx_14', 20.0))
            di_pos = float(row.get('di_pos', 20.0))
            di_neg = float(row.get('di_neg', 20.0))
            bb_h = float(row.get('bb_h', px))
            bb_l = float(row.get('bb_l', px))
            macd = float(row.get('macd', 0.0))
            macd_signal = float(row.get('macd_signal', 0.0))
            macd_bull = macd >= macd_signal
            atr = float(row.get('atr_14', float(df['close'].pct_change().rolling(14).std().iloc[i] * px)))
            buy_score = 0.0
            sell_score = 0.0
            if ema20 > ema50: buy_score += 0.25
            elif ema20 < ema50: sell_score += 0.25
            if macd_bull: buy_score += 0.2
            else: sell_score += 0.2
            if rsi <= 30: buy_score += 0.15
            elif rsi >= 70: sell_score += 0.15
            if adx >= 25 and di_pos > di_neg: buy_score += 0.2
            elif adx >= 25 and di_neg > di_pos: sell_score += 0.2
            if px <= bb_l: buy_score += 0.1
            elif px >= bb_h: sell_score += 0.1
            total = max(buy_score, sell_score)
            direction = 1 if buy_score > sell_score else -1
            if total < 0.92:
                continue
            trades += 1
            if direction == 1:
                sl = float(row.get('piv_low', df['low'].rolling(5).min().iloc[i])) - atr * 0.3
                tp = float(row.get('piv_high', df['high'].rolling(5).max().iloc[i])) if float(row.get('piv_high', df['high'].rolling(5).max().iloc[i])) > px else px + atr * 2.0
            else:
                sl = float(row.get('piv_high', df['high'].rolling(5).max().iloc[i])) + atr * 0.3
                tp = float(row.get('piv_low', df['low'].rolling(5).min().iloc[i])) if float(row.get('piv_low', df['low'].rolling(5).min().iloc[i])) < px else px - atr * 2.0
            # simulate next 50 bars
            hit = 0
            horizon = min(i+50, n-1)
            for j in range(i+1, horizon):
                h = float(df['high'].iloc[j])
                l = float(df['low'].iloc[j])
                if direction == 1:
                    if l <= sl:
                        hit = -1
                        ret_sum += (sl - px) / px
                        break
                    if h >= tp:
                        hit = 1
                        ret_sum += (tp - px) / px
                        break
                else:
                    if h >= sl:
                        hit = -1
                        ret_sum += (px - sl) / px
                        break
                    if l <= tp:
                        hit = 1
                        ret_sum += (px - tp) / px
                        break
            if hit == 1: wins += 1
            elif hit == -1: losses += 1
        win_rate = float(wins) / trades if trades else 0.0
        return { 'pair': pair, 'tf': tf, 'trades': trades, 'wins': wins, 'losses': losses, 'win_rate': win_rate, 'total_return': ret_sum }
    except Exception as e:
        return { 'pair': pair, 'tf': tf, 'error': str(e) }
