import os
import json
import zipfile
from pathlib import Path
import requests
import pandas as pd
import yfinance as yf

ROOT = Path('.')
RAW = ROOT / 'data' / 'raw'
PROC = ROOT / 'data' / 'processed'
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

def fetch_fred(series_id: str, dest_filename: Path):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=30)
    dest_filename.write_bytes(r.content)

def fetch_tradingeconomics_calendar(api_key: str, dest_filename: Path):
    if not api_key:
        return
    url = f"https://api.tradingeconomics.com/calendar?c={api_key}"
    r = requests.get(url, timeout=30)
    dest_filename.write_bytes(r.content)

def fetch_cftc_cot(year: int, dest_zip: Path):
    url = f"https://www.cftc.gov/files/dea/history/fut_disagg_xls_{year}.zip"
    r = requests.get(url, timeout=60)
    dest_zip.write_bytes(r.content)
    with zipfile.ZipFile(dest_zip, 'r') as z:
        z.extractall(RAW / f"cot_{year}")

def fetch_alternative_fng(dest_filename: Path):
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    r = requests.get(url, timeout=30)
    dest_filename.write_bytes(r.content)

def fetch_vix_yahoo(dest_filename: Path):
    url = "https://query1.finance.yahoo.com/v7/finance/download/%5EVIX?period1=0&period2=9999999999&interval=1d&events=history"
    r = requests.get(url, timeout=30)
    dest_filename.write_bytes(r.content)

def fetch_deribit_book_summary(currency: str, dest_filename: Path):
    url = f"https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={currency}"
    r = requests.get(url, timeout=30)
    dest_filename.write_bytes(r.content)

def fetch_yahoo_options(ticker: str, dest_filename: Path, max_exps: int = 5):
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        rows = []
        for e in exps[:max_exps]:
            chain = t.option_chain(e)
            c = chain.calls.copy()
            c["expiration"] = e
            rows.append(c[["contractSymbol","expiration","strike","lastPrice","impliedVolatility"]])
        if rows:
            pd.concat(rows).to_csv(dest_filename, index=False)
        else:
            dest_filename.write_text("")
    except Exception:
        dest_filename.write_text("")

def main():
    print("Fetching FRED yields...")
    fetch_fred("DGS10", RAW / "UST_10Y.csv")
    fetch_fred("DGS2", RAW / "UST_2Y.csv")

    print("Fetching VIX (Yahoo)...")
    fetch_vix_yahoo(RAW / "VIX.csv")

    print("Fetching Fear & Greed index...")
    fetch_alternative_fng(RAW / "fear_greed.json")

    print("Fetching Deribit book summary (BTC)...")
    fetch_deribit_book_summary("BTC", RAW / "deribit_book_summary_btc.json")

    print("Fetching AAPL option chains (sample)...")
    fetch_yahoo_options("AAPL", RAW / "AAPL_options_sample.csv")

    api_te = os.getenv("TRADINGECONOMICS_KEY", "")
    if api_te:
        print("Fetching TradingEconomics calendar...")
        fetch_tradingeconomics_calendar(api_te, RAW / "calendar.json")

    try:
        print("Fetching CFTC COT archive...")
        fetch_cftc_cot(2024, RAW / "COT_2024.zip")
    except Exception:
        pass

    print("ETL complete. Raw files in data/raw/")

if __name__ == "__main__":
    main()

