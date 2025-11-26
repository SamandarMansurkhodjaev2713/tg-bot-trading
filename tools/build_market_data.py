import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

try:
    import yfinance as yf
except Exception:
    yf = None


# CONFIG

BASE_DIR = r"c:\Users\sam4k\Documents\trae_projects\gitt"
HIST_DIR = os.path.join(BASE_DIR, "data", "histdata")
NEWS_DIR = os.path.join(BASE_DIR, "data", "news")
SENT_DIR = os.path.join(BASE_DIR, "data", "sentiment")

os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(NEWS_DIR, exist_ok=True)
os.makedirs(SENT_DIR, exist_ok=True)

PAIRS = ["EURUSD", "XAUUSD"]
STOOQ_SYMBOLS = {"EURUSD": "eurusd", "XAUUSD": "xauusd"}
YF_SYMBOLS = {"EURUSD": "EURUSD=X", "XAUUSD": "XAUUSD=X"}


# FUNCTIONS

def fetch_stooq_data(symbol: str, interval: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i={interval}"
    df = pd.read_csv(url, parse_dates=["Date"])  # may raise on network issues
    df = df.rename(
        columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    ).sort_values("timestamp")
    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_yf_data(pair: str, interval: str, period: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed")
    candidates = []
    if pair == "XAUUSD":
        candidates = ["XAUUSD=X", "GC=F"]
    elif pair == "EURUSD":
        candidates = ["EURUSD=X"]
    else:
        candidates = [YF_SYMBOLS.get(pair, pair)]
    data = None
    last_err = None
    for symbol in candidates:
        try:
            data = yf.download(symbol, interval=interval, period=period, auto_adjust=False, progress=False)
            if data is not None and len(data) > 0:
                break
        except Exception as e:
            last_err = e
            data = None
    if data is None or len(data) == 0:
        raise RuntimeError(f"yfinance returned empty for {pair} {interval} {period}")
    data = data.reset_index()
    # Flatten multi-index columns if any
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data.columns = data.columns.droplevel(0)
        except Exception:
            data.columns = [c[-1] if isinstance(c, tuple) else c for c in data.columns]
    if "Datetime" in data.columns:
        data = data.rename(columns={"Datetime": "timestamp"})
    if "Date" in data.columns:
        data = data.rename(columns={"Date": "timestamp"})
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    cols_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    data = data.rename(columns=cols_map)
    keep = ["timestamp", "open", "high", "low", "close", "volume"]
    for c in keep:
        if c not in data.columns:
            data[c] = 0.0
    # Coerce numerics and drop bad rows
    for c in ["open", "high", "low", "close", "volume"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data[keep]
    data = data.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
    data["timestamp"] = data["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return data


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


def ema(series: pd.Series, period: int = 14) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr_from_arrays(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> pd.Series:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    tr_series = pd.Series(tr)
    return tr_series.rolling(period).mean()


def generate_sentiment(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    close = np.asarray(df["close"], dtype=float).reshape(-1)
    ema_vals = ema(pd.Series(close), period).to_numpy()
    high = np.asarray(df["high"], dtype=float).reshape(-1)
    low = np.asarray(df["low"], dtype=float).reshape(-1)
    atr_vals = atr_from_arrays(high, low, close, period).to_numpy()
    atr_vals[atr_vals == 0] = np.nan
    sent = (close - ema_vals) / atr_vals
    sent = np.clip(np.nan_to_num(sent, nan=0.0), -1.0, 1.0)
    df["sentiment"] = sent
    return df[["timestamp", "sentiment"]]


def fetch_ff_news(start_dt: datetime, end_dt: datetime, symbols: list) -> pd.DataFrame:
    base_url = "https://www.forexfactory.com/calendar?week="
    news = []
    dt = start_dt
    while dt <= end_dt:
        week_url = base_url + dt.strftime("%Y%m%d")
        try:
            r = requests.get(week_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            soup = BeautifulSoup(r.content, "html.parser")
            rows = soup.select(".calendar__row")
            for row in rows:
                try:
                    impact_class = row.select_one(".impact")
                    impact = "low"
                    if impact_class:
                        cls = impact_class.get("class", [])
                        if any("high" in c for c in cls):
                            impact = "high"
                        elif any("medium" in c for c in cls):
                            impact = "medium"
                    cur_tag = row.select_one(".calendar__currency")
                    country = cur_tag.text.strip() if cur_tag else ""
                    time_tag = row.select_one(".calendar__time")
                    date_tag = row.select_one(".calendar__date")
                    if time_tag and date_tag:
                        time_text = time_tag.text.strip()
                        date_text = date_tag.text.strip()
                        try:
                            dt_news = datetime.strptime(f"{date_text} {time_text}", "%b %d %I:%M%p")
                        except Exception:
                            continue
                        dt_news = dt_news.replace(tzinfo=None) + timedelta(hours=5)
                        ts = dt_news.strftime("%Y-%m-%dT%H:%M:%SZ")
                        if country in ["USD", "EUR"] or country in symbols:
                            news.append([ts, impact, country])
                except Exception:
                    continue
        except Exception:
            pass
        dt += timedelta(days=7)
    df_news = pd.DataFrame(news, columns=["timestamp", "impact", "symbol"]).sort_values("timestamp")
    return df_news


# MAIN

def main():
    summary = []
    for pair in PAIRS:
        sym = STOOQ_SYMBOLS[pair]

        # 1H data
        try:
            df1h = fetch_stooq_data(sym, "60")
        except Exception:
            df1h = fetch_yf_data(pair, interval="60m", period="720d")
        if len(df1h) < 4000:
            raise RuntimeError(f"{pair} 1H data too short: {len(df1h)} rows")
        path1h = os.path.join(HIST_DIR, f"{pair}_1h.csv")
        save_csv(df1h, path1h)
        summary.append((f"{pair}_1h.csv", len(df1h), df1h["timestamp"].iloc[0], df1h["timestamp"].iloc[-1]))

        # 15m data
        try:
            df15m = fetch_stooq_data(sym, "15")
        except Exception:
            df15m = fetch_yf_data(pair, interval="15m", period="60d")
        if len(df15m) < 4000:
            raise RuntimeError(f"{pair} 15m data too short: {len(df15m)} rows")
        path15m = os.path.join(HIST_DIR, f"{pair}_15m.csv")
        save_csv(df15m, path15m)
        summary.append((f"{pair}_15m.csv", len(df15m), df15m["timestamp"].iloc[0], df15m["timestamp"].iloc[-1]))

        # News
        start_dt = datetime.strptime(df1h["timestamp"].iloc[0], "%Y-%m-%dT%H:%M:%SZ")
        end_dt = datetime.strptime(df1h["timestamp"].iloc[-1], "%Y-%m-%dT%H:%M:%SZ")
        df_news = fetch_ff_news(start_dt, end_dt, [pair])
        path_news = os.path.join(NEWS_DIR, f"{pair}_news.csv")
        save_csv(df_news, path_news)
        summary.append((f"{pair}_news.csv", len(df_news), df_news["timestamp"].iloc[0] if len(df_news) > 0 else "", df_news["timestamp"].iloc[-1] if len(df_news) > 0 else ""))

        # Sentiment
        df_sent = generate_sentiment(df1h)
        df_sent_head = df_sent.head(2000)
        ts_list = df_sent_head["timestamp"].astype(str).tolist()
        s_list = df_sent_head["sentiment"].astype(float).tolist()
        sent_list = [
            {"timestamp": t, "pair": pair, "sentiment": float(s)}
            for t, s in zip(ts_list, s_list)
        ]
        path_sent = os.path.join(SENT_DIR, f"{pair}_sentiment.json")
        with open(path_sent, "w", encoding="utf-8") as f:
            json.dump(sent_list, f, indent=2)
        summary.append((f"{pair}_sentiment.json", len(sent_list), sent_list[0]["timestamp"], sent_list[-1]["timestamp"]))

    print("\n=== SUMMARY ===")
    for name, n, t0, t1 in summary:
        print(f"{name}: {n} rows/events, {t0} → {t1}")


if __name__ == "__main__":
    main()
