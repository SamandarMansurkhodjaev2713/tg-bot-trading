from fastapi import APIRouter, Query
from ..services.quotes import get_ohlc
from tools.data_sources import DataManager
from ..features.indicators import compute_features
from ..models.training import analyze_signal
from ..models.open_source_ai import predict_pair
import os
import aiohttp

router = APIRouter()

@router.get("/analyze")
async def analyze(pair: str, tf: str = Query("15m"), window: int = Query(500)):
    try:
        df = get_ohlc(pair, tf, window)
    except Exception:
        df = None
    if df is None or len(df) < 50:
        dm = DataManager({})
        import pandas as pd
        import numpy as np
        data = await dm.get_merged_market_data(pair, tf, window)
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
    try:
        price = float(df["close"].iloc[-1])
    except Exception:
        price = float(rec.get("price", 0.0))
    try:
        steps_map = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}
        k = steps_map.get(tf, 24)
        prev = float(df["close"].iloc[-min(len(df)-1, k)])
        change_24h = float((price / prev - 1.0) * 100.0)
    except Exception:
        change_24h = 0.0
    rec.update({"price": price, "change_24h": change_24h})
    return rec


@router.get("/ai/predict")
async def ai_predict(pair: str, tf: str = Query("15m")):
    return await predict_pair(pair, tf)

@router.post("/ai/chat")
async def ai_chat(pair: str, tf: str, question: str = ""):
    pred = await predict_pair(pair, tf)
    prompt = (
        f"Пара: {pair} TF: {tf}. Сигнал: {pred.get('action','hold')}. "
        f"RSI: {pred.get('rsi',50):.1f}. ADX: {pred.get('adx',20):.1f}. "
        f"SL: {pred.get('sl','-')}. TP: {pred.get('tp','-')}. "
        f"Вопрос: {question}. Дай краткий профессиональный ответ с управлением риском и планом сделки."
    )
    key = os.getenv('OPENROUTER_API_KEY', '')
    if key:
        try:
            headers = { 'Authorization': f'Bearer {key}', 'Content-Type': 'application/json' }
            body = {
                'model': os.getenv('OPENROUTER_MODEL', 'meta-llama/llama-3.1-8b-instruct'),
                'messages': [ { 'role': 'user', 'content': prompt } ],
                'max_tokens': 300
            }
            async with aiohttp.ClientSession() as session:
                async with session.post('https://api.openrouter.ai/v1/chat/completions', json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    js = await resp.json()
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
