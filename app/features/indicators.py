import pandas as pd
import numpy as np
import ta

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    s = df.copy()
    s["sma_10"] = ta.trend.sma_indicator(s["close"], 10)
    s["sma_20"] = ta.trend.sma_indicator(s["close"], 20)
    s["sma_50"] = ta.trend.sma_indicator(s["close"], 50)
    s["ema_10"] = ta.trend.ema_indicator(s["close"], 10)
    s["ema_20"] = ta.trend.ema_indicator(s["close"], 20)
    s["ema_50"] = ta.trend.ema_indicator(s["close"], 50)
    s["rsi_14"] = ta.momentum.rsi(s["close"], 14)
    stoch = ta.momentum.StochasticOscillator(high=s["high"], low=s["low"], close=s["close"])
    s["stoch_k"] = stoch.stoch()
    s["stoch_d"] = stoch.stoch_signal()
    s["cci_20"] = ta.trend.cci(s["high"], s["low"], s["close"], 20)
    s["roc_10"] = ta.momentum.roc(s["close"], 10)
    s["trix_15"] = ta.trend.trix(s["close"], 15)
    bb = ta.volatility.BollingerBands(s["close"], 20, 2)
    s["bb_mavg"] = bb.bollinger_mavg()
    s["bb_h"] = bb.bollinger_hband()
    s["bb_l"] = bb.bollinger_lband()
    s["bb_perc"] = bb.bollinger_pband()
    s["bb_width"] = bb.bollinger_wband()
    kelt = ta.volatility.KeltnerChannel(s["high"], s["low"], s["close"]) 
    s["kc_h"] = kelt.keltner_channel_hband()
    s["kc_l"] = kelt.keltner_channel_lband()
    don = ta.volatility.DonchianChannel(s["high"], s["low"], s["close"]) 
    s["don_h"] = don.donchian_channel_hband()
    s["don_l"] = don.donchian_channel_lband()
    try:
        atr = ta.volatility.AverageTrueRange(high=s["high"], low=s["low"], close=s["close"], window=14)
        s["atr_14"] = atr.average_true_range()
    except Exception:
        s["atr_14"] = s["close"].pct_change().rolling(14).std() * s["close"].shift(1)
    s["vol_hist_20"] = s["close"].pct_change().rolling(20).std() * np.sqrt(252)
    adx = ta.trend.ADXIndicator(high=s["high"], low=s["low"], close=s["close"], window=14)
    s["adx_14"] = adx.adx()
    s["di_pos"] = adx.adx_pos()
    s["di_neg"] = adx.adx_neg()
    s["pivot"] = (s["high"] + s["low"] + s["close"]) / 3
    s["fib_382"] = s["pivot"] * 0.382
    s["fib_618"] = s["pivot"] * 0.618
    s["pin_bar"] = ((s["high"] - s["close"]).abs() > 2 * (s["open"] - s["close"]).abs()).astype(int)
    s["engulfing"] = ((s["close"] > s["open"]) & (s["open"].shift(1) > s["close"].shift(1)) & (s["close"] > s["open"].shift(1)) & (s["open"] < s["close"].shift(1))).astype(int)
    s["doji"] = ((s["close"] - s["open"]).abs() <= (s["high"] - s["low"]) * 0.1).astype(int)
    s["hammer"] = ((s["open"] > s["close"]) & ((s["open"] - s["low"]) > 2 * (s["open"] - s["close"])) & ((s["high"] - s["open"]) < (s["open"] - s["close"])) ).astype(int)
    s["shooting_star"] = ((s["close"] > s["open"]) & ((s["high"] - s["close"]) > 2 * (s["close"] - s["open"])) & ((s["open"] - s["low"]) < (s["close"] - s["open"])) ).astype(int)
    s["piv_high"] = s["high"].rolling(5).max()
    s["piv_low"] = s["low"].rolling(5).min()
    s["channel_up"] = (s["high"] > s["kc_h"]).astype(int)
    s["channel_down"] = (s["low"] < s["kc_l"]).astype(int)
    s["atr_pos_size"] = s["atr_14"]
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s
