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
import os
import requests

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

@router.post("/ai/chat")
def ai_chat(pair: str, tf: str, question: str = ""):
    from ..models.open_source_ai import predict_pair
    import asyncio
    pred = asyncio.run(predict_pair(pair, tf))
    prompt = f"Пара: {pair} TF: {tf}. Сигнал: {pred.get('action','hold')}. RSI: {pred.get('rsi',50):.1f}. ADX: {pred.get('adx',20):.1f}. SL: {pred.get('sl','-')}. TP: {pred.get('tp','-')}. Вопрос: {question}. Дай краткий профессиональный ответ с управлением риском и планом сделки."
    key = os.getenv('OPENROUTER_API_KEY', '')
    if key:
        try:
            headers = { 'Authorization': f'Bearer {key}', 'Content-Type': 'application/json' }
            body = {
                'model': os.getenv('OPENROUTER_MODEL', 'meta-llama/llama-3.1-8b-instruct'),
                'messages': [ { 'role': 'user', 'content': prompt } ],
                'max_tokens': 300
            }
            resp = requests.post('https://api.openrouter.ai/v1/chat/completions', json=body, headers=headers, timeout=20)
            js = resp.json()
            txt = js.get('choices',[{}])[0].get('message',{}).get('content','') or ''
            return { 'pair': pair, 'tf': tf, 'answer': txt, 'signal': pred }
        except Exception:
            pass
    lines = []
    act = pred.get('action','hold')
    rsi = float(pred.get('rsi',50))
    adx = float(pred.get('adx',20))
    sl = pred.get('sl','-')
    tp = pred.get('tp','-')
    lines.append(f"Сигнал: {act.upper()} | RSI {rsi:.1f} | ADX {adx:.1f}")
    if adx >= 25:
        lines.append("Тренд умеренный/сильный: работать по направлению сигнала.")
    else:
        lines.append("Тренд слабый: учитывать диапазон и подтверждение.")
    if rsi >= 70:
        lines.append("Перекупленность: подтверждение и частичный вход.")
    elif rsi <= 30:
        lines.append("Перепроданность: подтверждение и частичный вход.")
    lines.append(f"Уровни: SL {sl} | TP {tp}")
    lines.append("Риск: не более 1% на сделку, размер позиции по ATR.")
    if question:
        lines.append(f"Ответ: ориентируйся на структуру, избегай торговли рядом с важными новостями.")
    return { 'pair': pair, 'tf': tf, 'answer': "\n".join(lines), 'signal': pred }
