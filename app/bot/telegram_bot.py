import os
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, ConversationHandler, CallbackQueryHandler, MessageHandler, filters
import asyncio
import aiohttp
from ..utils.env import load_env

API_URL = "http://127.0.0.1:8000"

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

async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pair = context.args[0] if context.args else "XAU/USD"
    tf = context.args[1] if len(context.args)>1 else "15m"
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/analyze", params={"pair": pair, "tf": tf, "window": 500}) as r:
            d = await r.json()
            act = d.get("action","hold")
            size = d.get("size","-")
            sl = d.get("sl","-")
            tp = d.get("tp","-")
            inds = d.get("explanation",{}).get("indicators",{})
            rsi = float(inds.get("rsi", 50))
            adx = float(inds.get("adx", 20))
            msg = []
            msg.append(f"📊 {pair} {tf}")
            msg.append(f"Сигнал: {_map_action(act)}")
            msg.append(f"Размер позиции: {size}")
            msg.append(f"Уровни: SL {sl} | TP {tp}")
            msg.append(f"Индикаторы: RSI {rsi:.1f} | ADX {adx:.1f}")
            msg.append(_explain_indicators(rsi, adx))
            await update.message.reply_text("\n".join(msg))

async def cmd_news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pair = context.args[0] if context.args else "XAU/USD"
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/news", params={"pair": pair, "hours": 24}) as r:
            await update.message.reply_text(str(await r.json()))

async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pair = context.args[0] if context.args else "XAU/USD"
    tf = context.args[1] if len(context.args)>1 else "15m"
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/backtest", params={"pair": pair, "tf": tf, "start": "", "end": ""}) as r:
            await update.message.reply_text(str(await r.json()))

async def cmd_train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tf = context.args[0] if context.args else "15m"
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{API_URL}/train", params={"tf": tf}) as r:
            await update.message.reply_text(str(await r.json()))

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Бот готов!\n\nКоманды:\n/analyze <pair> <tf>\n/news <pair> <hours>\n/backtest <pair> <tf>\n/train <tf>\n/aitrader <pair> <tf>\n/aitrain <tf>")

async def cmd_aitrader(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pair = context.args[0] if context.args else "EUR/USD"
    tf = context.args[1] if len(context.args)>1 else "15m"
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/ai/predict", params={"pair": pair, "tf": tf}) as r:
            d = await r.json()
            act = d.get('action','hold')
            sl = d.get('sl','-')
            tp = d.get('tp','-')
            rsi = float(d.get('rsi',50))
            adx = float(d.get('adx',20))
            msg = []
            msg.append(f"🤖 {pair} {tf}")
            msg.append(f"Сигнал: {_map_action(act)}")
            msg.append(f"Уровни: SL {sl} | TP {tp}")
            msg.append(f"Индикаторы: RSI {rsi:.1f} | ADX {adx:.1f}")
            msg.append(_explain_indicators(rsi, adx))
            msg.append("Риск: ≤1% на сделку, размер позиции адаптируйте к ATR")
            await update.message.reply_text("\n".join(msg))
    await update.message.reply_text("Напишите вопрос по сделке или рынку — отвечу как трейдер.")
    context.user_data['ait_pair'] = pair
    context.user_data['ait_tf'] = tf
    return AIT_CHAT

async def cmd_aitrain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tf = context.args[0] if context.args else "15m"
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{API_URL}/ai/train", json=["EUR/USD","GBP/USD","USD/JPY","XAU/USD"], params={"tf": tf}) as r:
            await update.message.reply_text(str(await r.json()))

async def cmd_expert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pair = context.args[0] if context.args else "EUR/USD"
    tf = context.args[1] if len(context.args)>1 else "15m"
    question = " ".join(context.args[2:]) if len(context.args)>2 else ""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/ai/predict", params={"pair": pair, "tf": tf}) as r:
            d = await r.json()
        act = d.get('action','hold')
        sl = d.get('sl','-')
        tp = d.get('tp','-')
        rsi = d.get('rsi',50)
        adx = d.get('adx',20)
        guidance = []
        guidance.append(f"Сигнал: {act.upper()} | RSI {rsi:.1f} | ADX {adx:.1f}")
        if adx >= 25:
            guidance.append("Тренд силён, работать по направлению сигнала.")
        else:
            guidance.append("Тренд слабый, учитывай боковик и ложные пробои.")
        if rsi >= 70:
            guidance.append("Перекупленность: ищи откат перед покупками.")
        elif rsi <= 30:
            guidance.append("Перепроданность: ищи откат перед продажами.")
        guidance.append(f"Уровни: SL {sl} | TP {tp}")
        if question:
            guidance.append(f"Вопрос: {question}")
            guidance.append("Ответ: соблюдай риск 1% на сделку, позицию адаптируй к ATR; избегай торговли за 30 минут до/после важных новостей.")
        await update.message.reply_text("\n".join(guidance))

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

async def run_bot():
    load_env()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("news", cmd_news))
    app.add_handler(CommandHandler("backtest", cmd_backtest))
    app.add_handler(CommandHandler("train", cmd_train))
    conv2 = ConversationHandler(
        entry_points=[CommandHandler("aitrader", cmd_aitrader)],
        states={
            AIT_CHAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, aitrader_chat)],
        },
        fallbacks=[]
    )
    app.add_handler(conv2)
    app.add_handler(CommandHandler("aitrain", cmd_aitrain))
    app.add_handler(CommandHandler("expert", cmd_expert))
    add_analyse_menu(app)
    await app.initialize()
    await app.bot.set_my_commands([
        BotCommand("start", "Приветствие и список команд"),
        BotCommand("analyze", "Анализ пары и таймфрейма"),
        BotCommand("news", "Сводка новостей по паре"),
        BotCommand("backtest", "Бэктест и метрики"),
        BotCommand("train", "Обучение моделей по TF"),
        BotCommand("aitrader", "ИИ сигнал и совет"),
        BotCommand("aitrain", "Обучение ИИ моделей"),
        BotCommand("analyse", "Меню выбора пары и TF"),
    ])
    await app.run_polling()

SELECT_PAIR, SELECT_TF = range(2)

async def analyse_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pairs = [
        [InlineKeyboardButton("EUR/USD", callback_data="EUR/USD"), InlineKeyboardButton("GBP/USD", callback_data="GBP/USD")],
        [InlineKeyboardButton("USD/JPY", callback_data="USD/JPY"), InlineKeyboardButton("XAU/USD", callback_data="XAU/USD")],
        [InlineKeyboardButton("AUD/USD", callback_data="AUD/USD"), InlineKeyboardButton("NZD/USD", callback_data="NZD/USD")]
    ]
    await update.message.reply_text("Выберите пару:", reply_markup=InlineKeyboardMarkup(pairs))
    return SELECT_PAIR

async def analyse_pick_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    context.user_data['pair'] = q.data
    tfs = [[InlineKeyboardButton(x, callback_data=x) for x in ["15m","1h","4h","1d"]]]
    await q.edit_message_text("Выберите таймфрейм:")
    await q.message.reply_text("Таймфрейм:", reply_markup=InlineKeyboardMarkup(tfs))
    return SELECT_TF

async def analyse_pick_tf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    tf = q.data
    pair = context.user_data.get('pair', 'EUR/USD')
    async with aiohttp.ClientSession() as session:
        r = await session.get(f"{API_URL}/analyze", params={"pair": pair, "tf": tf, "window": 500})
        d = await r.json()
        act = d.get("action","hold")
        size = d.get("size","-")
        sl = d.get("sl","-")
        tp = d.get("tp","-")
        inds = d.get("explanation",{}).get("indicators",{})
        rsi = float(inds.get("rsi", 50))
        adx = float(inds.get("adx", 20))
        await q.edit_message_text(f"{pair} {tf} — анализ готов")
        msg = []
        msg.append(f"Сигнал: {_map_action(act)}")
        msg.append(f"Размер позиции: {size}")
        msg.append(f"Уровни: SL {sl} | TP {tp}")
        msg.append(f"Индикаторы: RSI {rsi:.1f} | ADX {adx:.1f}")
        msg.append(_explain_indicators(rsi, adx))
        await q.message.reply_text("\n".join(msg))
    return ConversationHandler.END

def add_analyse_menu(app):
    conv = ConversationHandler(
        entry_points=[CommandHandler("analyse", analyse_start)],
        states={
            SELECT_PAIR: [CallbackQueryHandler(analyse_pick_pair)],
            SELECT_TF: [CallbackQueryHandler(analyse_pick_tf)],
        },
        fallbacks=[]
    )
    app.add_handler(conv)
