from fastapi import APIRouter, Query
from ..services.quotes import get_ohlc
from tools.data_sources import DataManager
from ..features.indicators import compute_features
from ..models.training import analyze_signal, unified_analysis
from ..models.open_source_ai import predict_pair
import os
import aiohttp

router = APIRouter()

@router.get("/analyze")
async def analyze(pair: str, tf: str = Query("15m"), window: int = Query(500)):
    res = await unified_analysis(pair, tf, window)
    return res


@router.get("/ai/predict")
async def ai_predict(pair: str, tf: str = Query("15m")):
    return await unified_analysis(pair, tf, 600)

@router.post("/ai/chat")
async def ai_chat(pair: str, tf: str, question: str = ""):
    ua = await unified_analysis(pair, tf, 600)
    final = ua.get('final', {})
    ind = ua.get('indicators', {})
    prompt = (
        f"Пара: {pair} TF: {tf}. Сигнал: {final.get('direction','none')}. "
        f"RSI: {ind.get('rsi',50):.1f}. ADX: {ind.get('adx',20):.1f}. "
        f"SL: {final.get('sl','-')}. TP: {final.get('tp','-')}. "
        f"Вопрос: {question}. Дай краткий профессиональный ответ с логикой, уровнями (обоснованными) и управлением риском."
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
            return { 'pair': pair, 'tf': tf, 'answer': txt, 'signal': ua }
        except Exception:
            pass
    lines = []
    act = final.get('direction','none')
    rsi = float(ind.get('rsi',50))
    adx = float(ind.get('adx',20))
    sl = final.get('sl','-')
    tp = final.get('tp','-')
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
    return { 'pair': pair, 'tf': tf, 'answer': "\n".join(lines), 'signal': ua }
