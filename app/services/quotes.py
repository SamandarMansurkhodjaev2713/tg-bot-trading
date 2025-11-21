import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

PAIR_TICKERS = {
    "XAU/USD": "XAUUSD=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "USD/CHF": "CHF=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CAD": "CAD=X",
}

def _interval(tf: str) -> str:
    m = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "4h": "240m", "1d": "1d"}
    return m.get(tf, "15m")

def get_ohlc(pair: str, tf: str, window: int) -> pd.DataFrame:
    t = PAIR_TICKERS.get(pair)
    if not t:
        raise ValueError("pair")
    end = datetime.utcnow()
    start = end - timedelta(days=120)
    data = yf.download(t, start=start, end=end, interval=_interval(tf), progress=False)
    data = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"})
    data = data.dropna()
    if window and len(data) > window:
        data = data.tail(window)
    data.index = pd.to_datetime(data.index)
    return data