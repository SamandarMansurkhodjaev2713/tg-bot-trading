import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

PAIR_TICKERS = {
    "XAU/USD": "GC=F",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CAD": "USDCAD=X",
}

def _interval(tf: str) -> str:
    m = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "4h": "60m", "1d": "1d"}
    return m.get(tf, "15m")

def get_ohlc(pair: str, tf: str, window: int) -> pd.DataFrame:
    t = PAIR_TICKERS.get(pair)
    if not t:
        raise ValueError("pair")
    ticker = yf.Ticker(t)
    days = 5
    if tf in ["1m","5m","15m","1h","4h"]:
        if tf == "1m":
            days = min(7, max(1, int(window/60/24))) or 1
        elif tf == "5m":
            days = min(30, max(1, int(window/12/24))) or 7
        elif tf == "15m":
            days = min(60, max(1, int(window/4/24))) or 7
        elif tf in ["1h","4h"]:
            days = min(60, max(1, int(window/24))) or 7
    else:
        days = min(3650, max(1, int(window)))
    period = f"{days}d"
    data = ticker.history(period=period, interval=_interval(tf))
    data = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"})
    data = data.dropna()
    if window and len(data) > window:
        data = data.tail(window)
    data.index = pd.to_datetime(data.index)
    return data
