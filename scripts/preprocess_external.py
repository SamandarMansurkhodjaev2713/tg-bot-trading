from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path('data/raw')
PROC = Path('data/processed')
PROC.mkdir(parents=True, exist_ok=True)

def compute_indicators(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    df = df.copy()
    df['ema_12'] = df[price_col].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df[price_col].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    delta = df[price_col].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(com=13, adjust=False).mean()
    roll_down = down.ewm(com=13, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df = df.fillna(method='bfill').fillna(0)
    return df

def process_vix():
    path = RAW / 'VIX.csv'
    if path.exists():
        try:
            df = pd.read_csv(path)
        except Exception:
            return
        cols = {c.lower(): c for c in df.columns}
        ts_col = cols.get('date') or cols.get('timestamp') or list(df.columns)[0]
        close_col = cols.get('close') or 'Close' if 'Close' in df.columns else list(df.columns)[1] if len(df.columns) > 1 else None
        if ts_col:
            df['timestamp'] = pd.to_datetime(df[ts_col], errors='coerce')
        if close_col:
            df['close'] = pd.to_numeric(df[close_col], errors='coerce')
        if 'timestamp' in df.columns and 'close' in df.columns:
            df = df[['timestamp','close']].dropna()
            if not df.empty:
                df = compute_indicators(df, price_col='close')
                df.to_csv(PROC / 'VIX_processed.csv', index=False)

def process_aapl_options():
    path = RAW / 'AAPL_options_sample.csv'
    if path.exists():
        df = pd.read_csv(path)
        df['impliedVolatility'] = pd.to_numeric(df.get('impliedVolatility'), errors='coerce')
        df.to_csv(PROC / 'AAPL_options_processed.csv', index=False)

def main():
    process_vix()
    process_aapl_options()
    print('Preprocess complete. Processed files in data/processed/')

if __name__ == '__main__':
    main()
