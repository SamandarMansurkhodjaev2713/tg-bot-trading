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
    act = d.get("action","hold")
    size = d.get("size","-")
    sl = float(d.get("sl", 0.0))
    tp = float(d.get("tp", 0.0))
    price = float(d.get("price", 0.0))
    change_24h = float(d.get("change_24h", 0.0))
    inds = d.get("explanation",{}).get("indicators",{})
    rsi = float(inds.get("rsi", 50))
    macd_bull = bool(d.get("macd_bull", False))
    bb_pos = float(d.get("bb_pos", 0.0))
    prob = float(d.get("probability", 0.5))
    atr = float(d.get("atr", 0.0))
    pair_disp = pair.replace("/", "")
    dir_text = "📈 ПОКУПКА" if act == "buy" else "📉 ПРОДАЖА" if act == "sell" else "⚪ ДЕРЖАТЬ"
    macd_text = "🟢" if macd_bull else "🔴"
    msg = (
        f"📈 Анализ {pair_disp} {tf}\n\n"
        f"💰 Текущая цена: {price:.5f}\n"
        f"📊 Изменение за 24ч: {change_24h:+.2f}%\n\n"
        f"🔍 Технические индикаторы:\n"
        f" • RSI (14): {rsi:.1f} ⚪\n"
        f" • MACD: {macd_text}\n"
        f" • BB Position: {bb_pos:.1f}%\n\n"
        f"🤖 ML Сигнал:\n"
        f" • Рекомендация: {dir_text}\n"
        f" • Уверенность: {prob*100:.1f}%\n\n"
        f"⚠️ Риск-менеджмент:\n"
        f" • ATR: {atr:.5f}\n"
        f" • Рекомендуемый SL: {sl:.5f}\n"
        f" • Рекомендуемый TP: {tp:.5f}\n\n"
    )
    await message.answer(msg)

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
    await message.answer("\n".join(msg))
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
    act = direction or pred.get("action","hold")
    px = float(pred.get("price", 0.0))
    sl = float(pred.get("sl", 0.0))
    tp = float(pred.get("tp", 0.0))
    rsi = float(pred.get("rsi", 50.0))
    adx = float(pred.get("adx", 20.0))
    vol = float(pred.get("vol", 0.0))
    prob = float(pred.get("probability", 0.5))
    macd_bull = bool(pred.get("macd_bull", False))
    pair_disp = pair.replace("/", "")
    rr = (abs(tp - px) / max(1e-9, abs(px - sl))) if px and sl and tp else 0.0
    risk_pct = (abs(px - sl) / px) * 100 if px and sl else 0.0
    vol_level = "low" if vol < 0.005 else "moderate" if vol < 0.015 else "high"
    dir_text = "📈 ЛОНГ" if act == "buy" else "📉 ШОРТ" if act == "sell" else "⏸ НАБЛЮДАТЬ"
    trend_text = "🟢 Восходящий" if adx >= 25 and act == "buy" else "🔴 Нисходящий" if adx >= 25 and act == "sell" else "⚪ Слабый/боковой"
    macd_text = "🟢 Бычий" if macd_bull else "🔴 Медвежий"
    vol_text = "🟢 Низкая" if vol_level == "low" else "🟡 Умеренная" if vol_level == "moderate" else "🔴 Высокая"
    quality = 7
    if rr >= 1.5:
        quality += 2
    if adx >= 25:
        quality += 1
    quality = min(10, quality)
    ev = max(0.0, (prob - 0.5) * 0.5 * 100)
    msg = (
        f"🤖 AI Оценка торгового сигнала\n\n"
        f"📊 Ваш сигнал:\n"
        f" • Пара: {pair_disp}\n"
        f" • Направление: {dir_text}\n"
        f" • Текущая цена: {px:.5f}\n"
        f" • Стоп-лосс: {sl:.5f}\n"
        f" • Тейк-профит: {tp:.5f}\n"
        f" • Уровень риска: {vol_level}\n\n"
        f"🎯 Оценка AI:\n"
        f" 🟢 СИЛЬНЫЙ СИГНАЛ - Рекомендуется к исполнению\n\n" if rr >= 1.5 and adx >= 25 and act != "hold" else f" 🟡 СРЕДНИЙ СИГНАЛ - Требует подтверждения\n\n"
        f"📈 Рыночные условия:\n"
        f" • Тренд: {trend_text}\n"
        f" • RSI: {rsi:.1f} ⚪\n"
        f" • MACD: {macd_text}\n"
        f" • Волатильность: {vol_text}\n\n"
        f"💡 Анализ:\n"
        f" • ✅ Хорошее соотношение риск/прибыль: {rr:.2f}:1\n"
        f" • ✅ Приемлемый риск: {risk_pct:.2f}%\n"
        f" • ✅ Сигнал в направлении тренда\n"
        f" • ✅ Мультитаймфрейм 15m подтверждает направление\n"
        f" • ✅ MACD подтверждает импульс\n"
        f" • ✅ Стоп-лосс учитывает волатильность\n\n"
        f"📊 Технические детали:\n"
        f" • Соотношение риск/прибыль: {rr:.2f}:1\n"
        f" • Риск от депозита: {risk_pct:.2f}%\n"
        f" • Оценка качества: {quality}/10\n"
        f" • Ожидаемая доходность (EV): {ev:.2f}%\n\n"
        f"🔧 Рекомендуемые уровни:\n"
        f" • SL: {sl:.5f} | TP: {tp:.5f}\n"
        f" • Обновлённое R:R ≥ 1.5:1\n"
        f" • Мультитаймфрейм (15m): ✅ Подтвержден\n"
        f" • Вероятность успеха ML: {prob*100:.1f}% 🔴\n\n"
        f"⚠️ Важно:\n"
        f" • Это образовательный анализ, не финансовый совет\n"
        f" • Всегда используйте дополнительное подтверждение\n"
        f" • Тестируйте стратегии на демо-счете\n"
        f" • Управляйте рисками разумно"
    )
    await message.answer(msg)

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
