import os
import asyncio
import aiohttp
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from aiogram.filters import Command
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from app.utils.env import load_env

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

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

def _format_unified(res: dict) -> str:
    pair = res.get('pair','XAU/USD')
    tf = res.get('tf','15m')
    news = res.get('news',{})
    hist = res.get('history',{})
    sent = res.get('sentiment',{})
    fg = res.get('fear_greed',{})
    inst = res.get('institutional',{})
    ind = res.get('indicators',{})
    final = res.get('final',{})
    dir_txt = "BUY" if final.get('direction') == 'buy' else "SELL" if final.get('direction') == 'sell' else "NONE"
    lines = []
    lines.append(f"📌 {pair} {tf}")
    lines.append("1. Новости")
    lines.append(f" • Итог: {news.get('summary','')}\n • Влияние на XAUUSD: {news.get('influence','')}")
    lines.append("2. Исторические данные")
    lines.append(f" • Тренд: {hist.get('trend','')}\n • ATR: {hist.get('volatility_atr',0):.5f}\n • Уровни: L {hist.get('levels',{}).get('piv_low','-')} | H {hist.get('levels',{}).get('piv_high','-')}")
    lines.append("3. Сентимент рынка")
    lines.append(f" • Режим: {sent.get('risk_mode','')}\n • SP500 Δ: {sent.get('sp500_change',0.0):+.2%}\n • DXY Δ: {sent.get('dxy_change',0.0):+.2%}")
    lines.append("4. Индекс страха/жадности")
    lines.append(f" • Значение: {fg.get('value','n/a')}")
    lines.append("5. Сделки крупных игроков")
    cot = inst.get('cot')
    if cot:
        mm_net = float(cot.get('mm_net', 0) or 0)
        interp = "бычий" if mm_net > 0 else "медвежий" if mm_net < 0 else "нейтральный"
        lines.append(f" • COT GOLD: MM net {mm_net:+.0f} ({interp}) | дата {cot.get('report_date','')}")
    else:
        lines.append(" • Данные: n/a")
    lines.append("6. Индикаторы")
    lines.append(f" • EMA20: {ind.get('ema20','-'):.5f} | EMA50: {ind.get('ema50','-'):.5f}\n • RSI(14): {ind.get('rsi',50):.1f}\n • ADX(14): {ind.get('adx',20):.1f}\n • MACD: {'бычий' if ind.get('macd_bull', False) else 'медвежий'}")
    lines.append("7. Общий итог")
    lines.append(f" • Совпадение факторов: {res.get('final',{}).get('confidence',0.0):.2f}\n • Вывод: {dir_txt}")
    lines.append("8. Стоп-лосс (объяснение)")
    lines.append(f" • {final.get('sl','-')} — {final.get('sl_reason','')}")
    lines.append("9. Тейк-профит (объяснение)")
    lines.append(f" • {final.get('tp','-')} — {final.get('tp_reason','')}")
    lines.append("10. Финальный сигнал")
    lines.append(f" • {dir_txt}")
    return "\n".join(lines)

router = Router()
PAIRS = ["XAU/USD","EUR/USD","GBP/USD","USD/JPY","USD/CHF","AUD/USD","NZD/USD","USD/CAD"]
TF_ALLOWED = {"1m","5m","15m","30m","1h","4h","1d"}

def _norm(s: str) -> str:
    return s.lower().replace("/", "").replace("_", "")

def _find_pair(args: list[str]) -> str | None:
    m = { _norm(p): p for p in PAIRS }
    for a in args[1:]:
        key = _norm(a)
        if key in m:
            return m[key]
    for a in args[1:]:
        if "/" in a and a.upper() in PAIRS:
            return a.upper()
    return None

def _find_tf(args: list[str], default: str = "15m") -> str:
    for a in args[1:]:
        v = a.lower()
        if v in TF_ALLOWED:
            return v
    return default

class Analyse(StatesGroup):
    pick_pair = State()
    pick_tf = State()

class AITChat(StatesGroup):
    chat = State()

@router.message(Command("start"))
async def cmd_start(message: Message):
    text = (
        "🤖 Добро пожаловать в Forex AI Advisor!\n\n"
        "Я - ваш интеллектуальный помощник для торговли на форекс.\n\n"
        "📊 Доступные команды:\n"
        "• /analyze пара таймфрейм - Анализ валютной пары\n"
        "• /chatai сигнал - AI оценка вашего торгового сигнала\n"
        "• /aitrader запрос - Продвинутый AI трейдер с графиками\n\n"
        "💡 Примеры использования:\n"
        "• /analyze XAUUSD 1h - Анализ золота на 1 час\n"
        "• /chatai Хочу лонг XAUUSD со стопом 2650 и тейком 2720 - AI оценка сигнала\n"
        "• /aitrader Покажи график GBPUSD и дай рекомендации - Продвинутый AI анализ\n\n"
        "⚠️ Важно: Используйте только для образовательных целей!"
    )
    await message.answer(text)

@router.message(Command("analyze"))
async def cmd_analyze(message: Message):
    args = (message.text or "").split()
    pair = _find_pair(args)
    if not pair:
        await message.answer("пара не указана")
        return
    tf = _find_tf(args, "15m")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/analyze", params={"pair": pair, "tf": tf, "window": 500}) as r:
            ct = r.headers.get("Content-Type", "")
            if "application/json" in ct:
                d = await r.json()
            else:
                await message.answer("ошибка API")
                return
    await message.answer(_format_unified(d))

@router.message(Command("aitrader"))
async def cmd_aitrader(message: Message, state: FSMContext):
    args = (message.text or "").split()
    pair = _find_pair(args)
    if not pair:
        await message.answer("пара не указана")
        return
    tf = _find_tf(args, "15m")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/ai/predict", params={"pair": pair, "tf": tf}) as r:
            ct = r.headers.get("Content-Type", "")
            if "application/json" in ct:
                d = await r.json()
            else:
                await message.answer("ошибка API")
                return
    await message.answer(_format_unified(d))
    await message.answer("Напишите вопрос по сделке или рынку — отвечу как трейдер.")
    await state.update_data(ait_pair=pair, ait_tf=tf)
    await state.set_state(AITChat.chat)

@router.message(AITChat.chat)
async def aitrader_chat(message: Message, state: FSMContext):
    data = await state.get_data()
    pair = data.get('ait_pair', 'EUR/USD')
    tf = data.get('ait_tf', '15m')
    q = message.text or ""
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{API_URL}/ai/chat", params={"pair": pair, "tf": tf, "question": q}) as r:
            d = await r.json()
    txt = d.get('answer','')
    await message.answer(txt or "Готово.")
    await state.set_state(AITChat.chat)

@router.message(Command("chatai"))
async def cmd_chatai(message: Message, state: FSMContext):
    text = message.text or ""
    args = text.split()
    # detect pair in free text
    pairs = ["XAU/USD","EUR/USD","GBP/USD","USD/JPY","USD/CHF","AUD/USD","NZD/USD","USD/CAD"]
    lower = text.lower()
    pair = next((p for p in pairs if p.lower() in lower.replace(" ", "")), None)
    if not pair:
        await message.answer("пара не указана")
        return
    tf = "15m"
    # detect direction from text
    direction = None
    if any(w in lower for w in ["лонг","long","buy","покупка"]):
        direction = "buy"
    elif any(w in lower for w in ["шорт","short","sell","продажа"]):
        direction = "sell"
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/ai/predict", params={"pair": pair, "tf": tf}) as r:
            ct = r.headers.get("Content-Type", "")
            if "application/json" in ct:
                pred = await r.json()
            else:
                await message.answer("ошибка API")
                return
    await message.answer(_format_unified(pred))

async def run_bot():
    load_env()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    bot = Bot(token)
    dp = Dispatcher()
    dp.include_router(router)
    await bot.delete_webhook(drop_pending_updates=True)
    await bot.set_my_commands([
        BotCommand(command="analyze", description="Анализ пары и таймфрейма"),
        BotCommand(command="aitrader", description="ИИ сигнал и совет"),
        BotCommand(command="chatai", description="AI оценка вашего сигнала"),
    ])
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(run_bot())
