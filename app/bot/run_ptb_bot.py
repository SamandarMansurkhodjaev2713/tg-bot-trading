import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, ConversationHandler, MessageHandler, filters
import asyncio
import aiohttp
from ..utils.env import load_env

API_URL = "http://127.0.0.1:8000"

PAIR_MAP = {
    "XAU/USD": "XAU/USD",
    "XAUUSD": "XAU/USD",
    "EUR/USD": "EUR/USD",
    "EURUSD": "EUR/USD",
    "GBP/USD": "GBP/USD",
    "GBPUSD": "GBP/USD",
    "USD/JPY": "USD/JPY",
    "USDJPY": "USD/JPY",
    "USD/CHF": "USD/CHF",
    "USDCHF": "USD/CHF",
    "AUD/USD": "AUD/USD",
    "AUDUSD": "AUD/USD",
    "NZD/USD": "NZD/USD",
    "NZDUSD": "NZD/USD",
    "USD/CAD": "USD/CAD",
    "USDCAD": "USD/CAD",
}

def _normalize_pair(p: str) -> str:
    k = (p or "").strip().upper().replace(" ", "")
    return PAIR_MAP.get(k, "XAU/USD")

def _map_action(a: str) -> str:
    if a == "buy":
        return "Покупка"
    if a == "sell":
        return "Продажа"
    return "Наблюдать"

def _explain_indicators(rsi: float, adx: float) -> str:
    tips = []
    if adx >= 25:
        tips.append("Тренд сильный — работаем по направлению сигнала")
    else:
        tips.append("Тренд слабый — ждём подтверждения, учитываем диапазон")
    if rsi >= 70:
        tips.append("RSI>70: перекупленность — частичный вход или откат")
    elif rsi <= 30:
        tips.append("RSI<30: перепроданность — частичный вход или откат")
    else:
        tips.append("RSI нейтральный — ориентир по уровню и структуре")
    return "\n".join([f"• {t}" for t in tips])

def _format_detailed(d: dict, pair: str, tf: str, chat_answer: str | None = None) -> str:
    from datetime import datetime
    final = d.get('final', {})
    act = d.get('action', final.get('direction', 'none'))
    act_u = {
        'buy': 'BUY',
        'sell': 'SELL',
        'hold': 'HOLD',
        'none': 'OBSERVE'
    }.get(str(act).lower(), 'OBSERVE')
    inds = d.get('indicators', d.get('explanation', {}).get('indicators', {}))
    ema20 = inds.get('ema20')
    ema50 = inds.get('ema50')
    rsi = float(inds.get('rsi', d.get('rsi', 50)))
    adx = float(inds.get('adx', d.get('adx', 20)))
    macd_bull = bool(inds.get('macd_bull', False))
    bb_low = bool(inds.get('bb_low_touch', False))
    bb_high = bool(inds.get('bb_high_touch', False))
    news = d.get('news', {})
    sentiment = d.get('sentiment', {})
    institutional = d.get('institutional', {})
    conf = float(d.get('confidence', final.get('confidence', 0)) or 0)
    sl = final.get('sl', d.get('sl'))
    tp = final.get('tp', d.get('tp'))
    price = final.get('price', d.get('price'))
    strong = adx >= 25
    side = 'Вверх' if (macd_bull or (ema20 is not None and ema50 is not None and ema20 > ema50)) else ('Вниз' if (ema20 is not None and ema50 is not None and ema20 < ema50) else 'Нейтральное')
    trend = 'Сильный' if strong else ('Боковой' if adx < 20 else 'Слабый')
    reasons = []
    dr = final.get('direction_reason')
    if dr:
        reasons.append(dr)
    ex = d.get('explain')
    if ex:
        reasons.append(ex)
    if not reasons:
        if strong:
            reasons.append('Высокий ADX указывает на сильный рынок, работаем по направлению импульса.')
        else:
            reasons.append('ADX ниже порога, предпочтительна выжидательная тактика или работа в диапазоне.')
    ind_lines = []
    if ema20 is not None and ema50 is not None:
        ind_lines.append(f"EMA20: {ema20:.5f} — соотносится с EMA50 {ema50:.5f}, направление {('вверх' if ema20>ema50 else ('вниз' if ema20<ema50 else 'нейтрально'))}")
    ind_lines.append(f"RSI(14): {rsi:.1f} — {'перекупленность' if rsi>=70 else ('перепроданность' if rsi<=30 else 'нейтрально')} для входа")
    ind_lines.append(f"ADX(14): {adx:.1f} — {'сильный тренд' if strong else ('боковик' if adx<20 else 'слабый тренд')}")
    ind_lines.append(f"MACD: {'бычий импульс' if macd_bull else 'медвежий импульс'}")
    if bb_low:
        ind_lines.append("Bollinger Bands: касание нижней границы — потенциальная поддержка")
    if bb_high:
        ind_lines.append("Bollinger Bands: касание верхней границы — потенциальное сопротивление")
    ind_summary = "Усиление при согласованности EMA и MACD, подтверждение силой ADX; RSI уточняет момент входа."
    news_lines = []
    news_present = False
    influence = None
    if isinstance(news, dict):
        influence = news.get('influence')
        sm = news.get('summary')
        srcs = news.get('sources') or []
        if sm or srcs:
            news_present = True
        if sm:
            news_lines.append(f"Сводка: {sm}")
        tops = []
        for it in srcs[:2]:
            t = it.get('title')
            dmn = it.get('domain')
            if t:
                tops.append(f"{dmn}: {t}" if dmn else t)
        for x in tops:
            news_lines.append(f"• {x}")
    news_reco = "Избегать открытия за 30 минут до/после важных публикаций."
    rr_txt = "-"
    if isinstance(sl, (int,float)) and isinstance(tp, (int,float)) and isinstance(price, (int,float)):
        if act_u == 'BUY':
            risk = abs(price - float(sl))
            reward = abs(float(tp) - price)
        elif act_u == 'SELL':
            risk = abs(float(tp) - price)
            reward = abs(price - float(sl))
        else:
            risk = 0
            reward = 0
        rr_txt = f"{(reward/risk if risk>0 else 0):.2f}"
    sent_label = sentiment.get('label') if isinstance(sentiment, dict) else None
    whales = sentiment.get('whales') if isinstance(sentiment, dict) else None
    whale_note = None
    if whales:
        whale_note = f"Активность крупных игроков: {whales}"
    rec_summary = {
        'BUY': 'Купить',
        'SELL': 'Продать',
        'HOLD': 'Наблюдать',
        'OBSERVE': 'Наблюдать'
    }.get(act_u, 'Наблюдать')
    lines = []
    lines.append(f"🤖 Торговый анализ: {pair} | Таймфрейм: {tf}")
    lines.append(f"Дата и время анализа: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")
    lines.append("Сигнал")
    lines.append(f"Сигнал: {act_u}")
    lines.append(f"Причина сигнала: {'; '.join(reasons)}")
    lines.append("")
    lines.append("Тренд и структура рынка")
    lines.append(f"Тренд: {trend}")
    lines.append(f"Направление: {side}")
    lines.append(f"Объяснение: {('ADX подтверждает силу' if strong else 'ADX указывает на диапазон или слабый тренд')}, EMA20/EMA50 дают направление; MACD уточняет импульс")
    lines.append("")
    lines.append("Индикаторы")
    for l in ind_lines:
        lines.append(f"{l}")
    lines.append(f"Общее заключение по индикаторам: {ind_summary}")
    lines.append("")
    lines.append("Новости и события")
    lines.append(f"Важные новости: {'Есть' if news_present else 'Нет'}")
    lines.append(f"Влияние новостей: {influence or '-'}")
    if news_lines:
        for l in news_lines:
            lines.append(l)
    lines.append(f"Рекомендация: {news_reco}")
    lines.append("")
    lines.append("Уровни входа и выхода")
    lines.append(f"Stop Loss (SL): {sl if sl is not None else '-'}")
    lines.append(f"Take Profit (TP): {tp if tp is not None else '-'}")
    lines.append(f"Risk-Reward (RR): {rr_txt}")
    lines.append("Размер позиции: ≤1% на сделку, адаптировать к ATR")
    lines.append("")
    lines.append("Настроение рынка и сделки крупных игроков")
    lines.append(f"Общее настроение рынка: {sent_label or '-'}")
    if whale_note:
        lines.append(whale_note)
    lines.append(f"Влияние на стратегию: учитывать риск и силу тренда")
    lines.append("")
    lines.append("Итоговая рекомендация")
    lines.append(f"Резюме: {rec_summary}")
    lines.append(f"Главные причины: тренд {trend}, индикаторы RSI/EMA/ADX, новости {influence or '-'}")
    lines.append("Риск: ≤1% на сделку")
    if chat_answer:
        lines.append("")
        lines.append(f"Ответ AI: {chat_answer}")
    return "\n".join(lines)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Бот запущен. Команды:\n"
        "• /aitrader <пара> <таймфрейм> — AI сигнал\n"
        "• /chatai <вопрос> — AI объяснение"
    )

async def cmd_aitrader(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pair = _normalize_pair(context.args[0]) if context.args else "EUR/USD"
    tf = context.args[1] if len(context.args)>1 else "15m"
    try:
        d = None
        async with aiohttp.ClientSession() as session:
            for delay in [0.0, 0.8, 1.6]:
                if delay:
                    await asyncio.sleep(delay)
                try:
                    async with session.get(f"{API_URL}/ai/predict", params={"pair": pair, "tf": tf}) as r:
                        if r.status == 200:
                            d = await r.json()
                            break
                except Exception:
                    continue
            if not d:
                async with session.get(f"{API_URL}/analyze", params={"pair": pair, "tf": tf, "window": 500}) as r:
                    if r.status == 200:
                        d = await r.json()
        if not d:
            cached = context.user_data.get('last_ait', {})
            if cached:
                d = cached
            else:
                raise RuntimeError("no_response")
        txt = _format_detailed(d, pair, tf)
        await update.message.reply_text(txt)
        context.user_data['last_ait'] = d
        context.user_data['ait_pair'] = pair
        context.user_data['ait_tf'] = tf
    except Exception:
        await update.message.reply_text("Ошибка анализа. Использую последний успешный результат или попробуйте /aitrader EURUSD 1h.")
        cached = context.user_data.get('last_ait', {})
        if cached:
            await update.message.reply_text(_format_detailed(cached, pair, tf))

async def cmd_chatai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pair = context.user_data.get('ait_pair', 'EUR/USD')
    tf = context.user_data.get('ait_tf', '15m')
    question = " ".join(context.args) if context.args else "Объясни причины решения и риск-менеджмент"
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{API_URL}/ai/chat", params={"pair": pair, "tf": tf, "question": question}) as r:
            ans = await r.json()
        async with session.get(f"{API_URL}/ai/predict", params={"pair": pair, "tf": tf}) as r2:
            base = await r2.json()
    txt = ans.get('answer','')
    await update.message.reply_text(_format_detailed(base, pair, tf, txt))

AIT_CHAT = 100
async def aitrader_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.message.text
    pair = context.user_data.get('ait_pair', 'EUR/USD')
    tf = context.user_data.get('ait_tf', '15m')
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{API_URL}/ai/chat", params={"pair": pair, "tf": tf, "question": q}) as r:
            d = await r.json()
        txt = d.get('answer','')
        await update.message.reply_text(txt or "Готово.")
    return AIT_CHAT

def run_bot():
    load_env()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    conv = ConversationHandler(
        entry_points=[CommandHandler("aitrader", cmd_aitrader)],
        states={
            AIT_CHAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, aitrader_chat)],
        },
        fallbacks=[]
    )
    app.add_handler(conv)
    app.add_handler(CommandHandler("chatai", cmd_chatai))
    app.run_polling()
