from fastapi import APIRouter, Query
from typing import Optional
from ..services.quotes import get_ohlc
from data_sources import DataManager
from . import routes
from ..features.indicators import compute_features
from ..models.training import analyze_signal, train_models
from ..backtest.backtester import run_backtest
from ..services.news import news_summary
from ..models.open_source_ai import train_pairs, predict_pair

router = APIRouter()

@router.get("/analyze")
def analyze(pair: str, tf: str = Query("15m"), window: int = Query(500)):
    try:
        df = get_ohlc(pair, tf, window)
    except Exception:
        df = None
    if df is None or len(df) < 50:
        dm = DataManager({})
        import pandas as pd
        import numpy as np
        import asyncio
        data = asyncio.run(dm.get_merged_market_data(pair, tf, window))
        if data:
            d = pd.DataFrame(data)
            d['timestamp'] = pd.to_datetime(d['timestamp'])
            d = d.sort_values('timestamp').set_index('timestamp')
            df = d
        else:
            idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=window, freq='15min' if tf=='15m' else '1h')
            base = 100.0
            rnd = np.random.normal(0, 0.1, size=len(idx)).cumsum()
            close = base + rnd
            openp = close * (1 - np.random.normal(0, 0.001, size=len(idx)))
            high = np.maximum(openp, close) * (1 + np.random.uniform(0, 0.002, size=len(idx)))
            low = np.minimum(openp, close) * (1 - np.random.uniform(0, 0.002, size=len(idx)))
            vol = np.random.randint(500, 1500, size=len(idx))
            df = pd.DataFrame({"open": openp, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
    feats = compute_features(df)
    rec = analyze_signal(pair, tf, df, feats)
    return rec

@router.post("/train")
def train(pairs: Optional[list[str]] = None, tf: str = Query("15m")):
    res = train_models(pairs or ["XAU/USD","EUR/USD","GBP/USD","USD/JPY","USD/CHF","AUD/USD","NZD/USD","USD/CAD"], tf)
    return res

@router.get("/backtest")
def backtest(pair: str, tf: str, start: str, end: str):
    report = run_backtest(pair, tf, start, end)
    return report

@router.get("/news")
def news(pair: str, hours: int = Query(24)):
    return news_summary(pair, hours)

@router.post("/ai/train")
async def ai_train(pairs: Optional[list[str]] = None, tf: str = Query("15m")):
    return await train_pairs(pairs or ["XAU/USD","EUR/USD","GBP/USD","USD/JPY","USD/CHF","AUD/USD","NZD/USD","USD/CAD"], tf)

@router.get("/ai/predict")
async def ai_predict(pair: str, tf: str = Query("15m")):
    return await predict_pair(pair, tf)
