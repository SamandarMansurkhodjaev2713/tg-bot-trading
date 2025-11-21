import os
from telegram import Update, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import asyncio
import httpx
from ..utils.env import load_env

API_URL = "http://127.0.0.1:8000"

async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pair = context.args[0] if context.args else "XAU/USD"
    tf = context.args[1] if len(context.args)>1 else "15m"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(f"{API_URL}/analyze", params={"pair": pair, "tf": tf, "window": 500})
        await update.message.reply_text(str(r.json()))

async def cmd_news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pair = context.args[0] if context.args else "XAU/USD"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(f"{API_URL}/news", params={"pair": pair, "hours": 24})
        await update.message.reply_text(str(r.json()))

async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pair = context.args[0] if context.args else "XAU/USD"
    tf = context.args[1] if len(context.args)>1 else "15m"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(f"{API_URL}/backtest", params={"pair": pair, "tf": tf, "start": "", "end": ""})
        await update.message.reply_text(str(r.json()))

async def cmd_train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tf = context.args[0] if context.args else "15m"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{API_URL}/train", params={"tf": tf})
        await update.message.reply_text(str(r.json()))

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Бот готов. Команды: /analyze <pair> <tf>, /news <pair> <hours>, /backtest <pair> <tf>, /train <tf>.")

async def run_bot():
    load_env()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("news", cmd_news))
    app.add_handler(CommandHandler("backtest", cmd_backtest))
    app.add_handler(CommandHandler("train", cmd_train))
    await app.initialize()
    await app.bot.set_my_commands([
        BotCommand("start", "Приветствие и список команд"),
        BotCommand("analyze", "Анализ пары и таймфрейма"),
        BotCommand("news", "Сводка новостей по паре"),
        BotCommand("backtest", "Бэктест и метрики"),
        BotCommand("train", "Обучение моделей по TF"),
    ])
    await app.run_polling()