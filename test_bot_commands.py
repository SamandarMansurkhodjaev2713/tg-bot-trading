import asyncio
from typing import List, Dict
from datetime import datetime, timedelta

from simple_forex_bot import ForexBot


class DummyMessage:
    def __init__(self):
        self.outputs: List[str] = []

    async def reply_text(self, text, **kwargs):
        self.outputs.append(text)


class DummyUpdate:
    def __init__(self):
        self.message = DummyMessage()


class DummyContext:
    def __init__(self, args: List[str]):
        self.args = args


def make_quotes(n: int = 200) -> List[Dict]:
    quotes = []
    base = 1.1
    start = datetime(2025, 1, 1, 0, 0, 0)
    for i in range(n):
        p = base + i * 0.0005
        ts = (start + timedelta(minutes=i)).strftime('%Y-%m-%d %H:%M:%S')
        quotes.append({
            'timestamp': ts,
            'open': p - 0.0003,
            'high': p + 0.0006,
            'low': p - 0.0006,
            'close': p,
            'volume': 1000 + i
        })
    return quotes


async def run_case(bot: ForexBot, text: str) -> List[str]:
    upd = DummyUpdate()
    ctx = DummyContext(text.split()[1:])
    await bot.aitrader_command(upd, ctx)
    return upd.message.outputs


async def main():
    bot = ForexBot(token="TEST")

    def mock_get_quotes(pair: str, timeframe: str, limit: int = 100):
        return make_quotes(limit)

    bot.get_quotes = mock_get_quotes

    cases = [
        "/aitrader Проанализируй XAUUSD на вход в лонг",
        "/aitrader Дай рекомендации по EURUSD",
        "/aitrader Сигнал на BTCUSD с минимальным риском",
        "/aitrader Анализ USOIL"
    ]

    results = {}
    for c in cases:
        outputs = await run_case(bot, c)
        results[c] = outputs

    ok = True
    for k, v in results.items():
        if not v or not any("Advanced AI Trader Analysis" in o for o in v):
            ok = False

    print("OK" if ok else "FAIL")
    for k, v in results.items():
        print("CASE:", k)
        for o in v:
            print(o.splitlines()[0])

    # Test /chatai with BTC short
    upd = DummyUpdate()
    ctx = DummyContext(["Хочу", "шорт", "BTC", "со", "стопом", "87505", "и", "тейком", "84800"])
    bot.get_quotes = mock_get_quotes
    await bot.chatai_command(upd, ctx)
    chatai_output = "\n".join(upd.message.outputs)
    if "Не удалось распознать валютную пару" in chatai_output:
        print("CHAITAI FAIL")
    else:
        print("CHAITAI OK")


if __name__ == "__main__":
    asyncio.run(main())