from fastapi import APIRouter, Query
from typing import Optional
from ..services.quotes import get_ohlc
from ..features.indicators import compute_features
from ..models.training import analyze_signal, train_models
from ..backtest.backtester import run_backtest
from ..services.news import news_summary

router = APIRouter()

@router.get("/analyze")
def analyze(pair: str, tf: str = Query("15m"), window: int = Query(500)):
    df = get_ohlc(pair, tf, window)
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