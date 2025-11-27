import numpy as np
import math
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

class AdvancedTechnicalIndicators:
    """Advanced technical indicators for professional trading analysis"""
    
    def __init__(self):
        self.cache = {}
    
    def adx_dmi(self, quotes: List[Dict], period: int = 14) -> Dict[str, float]:
        """Calculate ADX and DMI indicators"""
        if len(quotes) < period + 1:
            return {'adx': 0, 'plus_di': 0, 'minus_di': 0, 'trend_strength': 0}
        
        highs = [q['high'] for q in quotes]
        lows = [q['low'] for q in quotes]
        closes = [q['close'] for q in quotes]
        
        tr_values = []
        plus_dm_values = []
        minus_dm_values = []
        
        for i in range(1, len(quotes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_values.append(tr)
            
            plus_dm = max(highs[i] - highs[i-1], 0) if highs[i] - highs[i-1] > lows[i-1] - lows[i] else 0
            minus_dm = max(lows[i-1] - lows[i], 0) if lows[i-1] - lows[i] > highs[i] - highs[i-1] else 0
            
            plus_dm_values.append(plus_dm)
            minus_dm_values.append(minus_dm)
        
        # Calculate smoothed values
        tr_smooth = sum(tr_values[:period])
        plus_dm_smooth = sum(plus_dm_values[:period])
        minus_dm_smooth = sum(minus_dm_values[:period])
        
        plus_di = 100 * plus_dm_smooth / tr_smooth if tr_smooth > 0 else 0
        minus_di = 100 * minus_dm_smooth / tr_smooth if tr_smooth > 0 else 0
        
        # Calculate DX and ADX
        dx_values = []
        for i in range(period, len(tr_values)):
            tr_smooth = tr_smooth - tr_smooth/period + tr_values[i]
            plus_dm_smooth = plus_dm_smooth - plus_dm_smooth/period + plus_dm_values[i]
            minus_dm_smooth = minus_dm_smooth - minus_dm_smooth/period + minus_dm_values[i]
            
            plus_di = 100 * plus_dm_smooth / tr_smooth if tr_smooth > 0 else 0
            minus_di = 100 * minus_dm_smooth / tr_smooth if tr_smooth > 0 else 0
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
            dx_values.append(dx)
        
        adx = sum(dx_values[:period]) / period if dx_values else 0
        for dx in dx_values[period:]:
            adx = (adx * (period - 1) + dx) / period
        
        trend_strength = min(adx / 25, 1.0)  # Normalize to 0-1
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'trend_strength': trend_strength
        }
    
    def aroon(self, quotes: List[Dict], period: int = 25) -> Dict[str, float]:
        """Calculate Aroon indicator"""
        if len(quotes) < period:
            return {'aroon_up': 0, 'aroon_down': 0, 'aroon_oscillator': 0}
        
        highs = [q['high'] for q in quotes]
        lows = [q['low'] for q in quotes]
        
        recent_high_idx = len(highs) - 1 - highs[-period:].index(max(highs[-period:]))
        recent_low_idx = len(lows) - 1 - lows[-period:].index(min(lows[-period:]))
        
        aroon_up = 100 * (period - (len(highs) - 1 - recent_high_idx)) / period
        aroon_down = 100 * (period - (len(lows) - 1 - recent_low_idx)) / period
        aroon_oscillator = aroon_up - aroon_down
        
        return {
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_oscillator
        }
    
    def supertrend(self, quotes: List[Dict], period: int = 10, multiplier: float = 3.0) -> Dict[str, float]:
        """Calculate Supertrend indicator"""
        if len(quotes) < period:
            return {'supertrend': 0, 'direction': 0, 'atr': 0}
        
        highs = [q['high'] for q in quotes]
        lows = [q['low'] for q in quotes]
        closes = [q['close'] for q in quotes]
        
        # Calculate ATR
        atr_values = []
        for i in range(1, len(quotes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            atr_values.append(tr)
        
        atr = sum(atr_values[:period]) / period if atr_values else 0
        for i in range(period, len(atr_values)):
            atr = (atr * (period - 1) + atr_values[i]) / period
        
        # Calculate Supertrend
        basic_upper_band = (highs[-1] + lows[-1]) / 2 + multiplier * atr
        basic_lower_band = (highs[-1] + lows[-1]) / 2 - multiplier * atr
        
        final_upper_band = basic_upper_band
        final_lower_band = basic_lower_band
        
        if len(quotes) > 1:
            prev_close = closes[-2]
            if prev_close > basic_upper_band:
                final_upper_band = basic_upper_band
            else:
                final_upper_band = min(basic_upper_band, (highs[-2] + lows[-2]) / 2 + multiplier * atr)
            
            if prev_close < basic_lower_band:
                final_lower_band = basic_lower_band
            else:
                final_lower_band = max(basic_lower_band, (highs[-2] + lows[-2]) / 2 - multiplier * atr)
        
        current_close = closes[-1]
        direction = 1 if current_close > final_upper_band else -1 if current_close < final_lower_band else 0
        supertrend_value = final_lower_band if direction == 1 else final_upper_band
        
        return {
            'supertrend': supertrend_value,
            'direction': direction,
            'atr': atr
        }
    
    def keltner_channels(self, quotes: List[Dict], period: int = 20, multiplier: float = 2.0) -> Dict[str, float]:
        """Calculate Keltner Channels"""
        if len(quotes) < period:
            return {'upper': 0, 'middle': 0, 'lower': 0, 'width': 0}
        
        closes = [q['close'] for q in quotes]
        highs = [q['high'] for q in quotes]
        lows = [q['low'] for q in quotes]
        
        # Calculate EMA
        ema = self._calculate_ema(closes, period)
        
        # Calculate ATR for bands
        atr_values = []
        for i in range(1, len(quotes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            atr_values.append(tr)
        
        atr = sum(atr_values[-period:]) / period if atr_values else 0
        
        upper_band = ema + multiplier * atr
        lower_band = ema - multiplier * atr
        width = upper_band - lower_band
        
        return {
            'upper': upper_band,
            'middle': ema,
            'lower': lower_band,
            'width': width
        }
    
    def donchian_channels(self, quotes: List[Dict], period: int = 20) -> Dict[str, float]:
        """Calculate Donchian Channels"""
        if len(quotes) < period:
            return {'upper': 0, 'middle': 0, 'lower': 0, 'width': 0}
        
        highs = [q['high'] for q in quotes]
        lows = [q['low'] for q in quotes]
        
        upper_band = max(highs[-period:])
        lower_band = min(lows[-period:])
        middle_band = (upper_band + lower_band) / 2
        width = upper_band - lower_band
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band,
            'width': width
        }
    
    def hurst_exponent(self, quotes: List[Dict], period: int = 100) -> float:
        """Calculate Hurst exponent for market regime detection"""
        if len(quotes) < period:
            return 0.5
        
        closes = [q['close'] for q in quotes[-period:]]
        
        # Calculate returns
        returns = []
        for i in range(1, len(closes)):
            returns.append(math.log(closes[i] / closes[i-1]))
        
        if len(returns) < 10:
            return 0.5
        
        # Calculate cumulative deviation
        mean_return = sum(returns) / len(returns)
        cumulative_dev = []
        cumulative_sum = 0
        
        for r in returns:
            cumulative_sum += r - mean_return
            cumulative_dev.append(cumulative_sum)
        
        # Calculate R/S statistic
        R = max(cumulative_dev) - min(cumulative_dev)
        S = math.sqrt(sum((r - mean_return) ** 2 for r in returns) / len(returns))
        
        if S == 0:
            return 0.5
        
        rs = R / S
        
        # Hurst exponent approximation
        hurst = math.log(rs) / math.log(len(returns)) if rs > 0 else 0.5
        
        return max(0, min(1, hurst))  # Clamp to [0, 1]
    
    def stochastic(self, quotes: List[Dict], period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Dict[str, float]:
        """Calculate Stochastic oscillator"""
        if len(quotes) < period:
            return {'k': 50, 'd': 50}
        
        closes = [q['close'] for q in quotes]
        highs = [q['high'] for q in quotes]
        lows = [q['low'] for q in quotes]
        
        current_close = closes[-1]
        period_high = max(highs[-period:])
        period_low = min(lows[-period:])
        
        if period_high == period_low:
            k = 50
        else:
            k = 100 * (current_close - period_low) / (period_high - period_low)
        
        # Calculate %D (smoothed %K)
        k_values = []
        for i in range(smooth_k - 1, len(closes)):
            start_idx = max(0, i-period+1)
            period_high = max(highs[start_idx:i+1])
            period_low = min(lows[start_idx:i+1])
            if period_high == period_low:
                k_val = 50
            else:
                k_val = 100 * (closes[i] - period_low) / (period_high - period_low)
            k_values.append(k_val)
        
        if len(k_values) >= smooth_d:
            d = sum(k_values[-smooth_d:]) / smooth_d
        else:
            d = k
        
        return {'k': k, 'd': d}
    
    def cci(self, quotes: List[Dict], period: int = 20) -> float:
        """Calculate Commodity Channel Index"""
        if len(quotes) < period:
            return 0
        
        highs = [q['high'] for q in quotes]
        lows = [q['low'] for q in quotes]
        closes = [q['close'] for q in quotes]
        
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        
        sma_tp = sum(typical_prices[-period:]) / period
        mean_deviation = sum(abs(tp - sma_tp) for tp in typical_prices[-period:]) / period
        
        if mean_deviation == 0:
            return 0
        
        cci = (typical_prices[-1] - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    def heikin_ashi(self, quotes: List[Dict]) -> Dict[str, List[float]]:
        """Calculate Heikin Ashi candlesticks for trend smoothing"""
        if len(quotes) < 2:
            return {'open': [q['open'] for q in quotes], 'high': [q['high'] for q in quotes], 
                   'low': [q['low'] for q in quotes], 'close': [q['close'] for q in quotes]}
        
        ha_open = []
        ha_high = []
        ha_low = []
        ha_close = []
        
        # First HA candlestick
        ha_open.append((quotes[0]['open'] + quotes[0]['close']) / 2)
        ha_close.append((quotes[0]['open'] + quotes[0]['high'] + quotes[0]['low'] + quotes[0]['close']) / 4)
        ha_high.append(max(quotes[0]['high'], ha_open[0], ha_close[0]))
        ha_low.append(min(quotes[0]['low'], ha_open[0], ha_close[0]))
        
        # Calculate remaining HA values
        for i in range(1, len(quotes)):
            prev_ha_open = ha_open[i-1]
            prev_ha_close = ha_close[i-1]
            
            current_close = (quotes[i]['open'] + quotes[i]['high'] + quotes[i]['low'] + quotes[i]['close']) / 4
            current_open = (prev_ha_open + prev_ha_close) / 2
            current_high = max(quotes[i]['high'], current_open, current_close)
            current_low = min(quotes[i]['low'], current_open, current_close)
            
            ha_close.append(current_close)
            ha_open.append(current_open)
            ha_high.append(current_high)
            ha_low.append(current_low)
        
        return {'open': ha_open, 'high': ha_high, 'low': ha_low, 'close': ha_close}
    
    def vwap(self, quotes: List[Dict]) -> List[float]:
        """Calculate Volume Weighted Average Price (VWAP) for intraday levels"""
        if len(quotes) == 0:
            return []
        
        vwap_values = []
        cumulative_volume_price = 0
        cumulative_volume = 0
        
        for quote in quotes:
            # Use typical price * volume if volume is available, otherwise use close price
            typical_price = (quote['high'] + quote['low'] + quote['close']) / 3
            volume = quote.get('volume', 1)  # Default to 1 if volume not available
            
            cumulative_volume_price += typical_price * volume
            cumulative_volume += volume
            
            if cumulative_volume > 0:
                vwap_values.append(cumulative_volume_price / cumulative_volume)
            else:
                vwap_values.append(typical_price)
        
        return vwap_values
    
    def mfi(self, quotes: List[Dict], period: int = 14) -> float:
        """Calculate Money Flow Index"""
        if len(quotes) < period:
            return 50
        
        highs = [q['high'] for q in quotes]
        lows = [q['low'] for q in quotes]
        closes = [q['close'] for q in quotes]
        volumes = [q.get('volume', 1) for q in quotes]
        
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        
        positive_money_flow = 0
        negative_money_flow = 0
        
        for i in range(1, len(typical_prices)):
            if typical_prices[i] > typical_prices[i-1]:
                positive_money_flow += typical_prices[i] * volumes[i]
            elif typical_prices[i] < typical_prices[i-1]:
                negative_money_flow += typical_prices[i] * volumes[i]
        
        if negative_money_flow == 0:
            return 100 if positive_money_flow > 0 else 50
        
        money_ratio = positive_money_flow / negative_money_flow
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    def obv(self, quotes: List[Dict]) -> float:
        """Calculate On-Balance Volume"""
        if len(quotes) < 2:
            return 0
        
        closes = [q['close'] for q in quotes]
        volumes = [q.get('volume', 1) for q in quotes]
        
        obv = 0
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv += volumes[i]
            elif closes[i] < closes[i-1]:
                obv -= volumes[i]
        
        return obv
    
    def ichimoku(self, quotes: List[Dict]) -> Dict[str, float]:
        """Calculate Ichimoku Cloud"""
        if len(quotes) < 52:
            return {'tenkan': 0, 'kijun': 0, 'senkou_a': 0, 'senkou_b': 0, 'chikou': 0}
        
        highs = [q['high'] for q in quotes]
        lows = [q['low'] for q in quotes]
        closes = [q['close'] for q in quotes]
        
        # Tenkan-sen (9-period)
        tenkan_high = max(highs[-9:])
        tenkan_low = min(lows[-9:])
        tenkan = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (26-period)
        kijun_high = max(highs[-26:])
        kijun_low = min(lows[-26:])
        kijun = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (26 periods ahead)
        senkou_a = (tenkan + kijun) / 2
        
        # Senkou Span B (52 periods, 26 ahead)
        senkou_b = (max(highs[-52:-26]) + min(lows[-52:-26])) / 2 if len(quotes) >= 52 else kijun
        
        # Chikou Span (26 periods back)
        chikou = closes[-27] if len(closes) >= 27 else closes[0]
        
        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou,
            'cloud_top': max(senkou_a, senkou_b),
            'cloud_bottom': min(senkou_a, senkou_b)
        }
    
    def detect_divergences(self, quotes: List[Dict], indicator_values: List[float], 
                          lookback: int = 20) -> Dict[str, bool]:
        """Detect divergences between price and indicator"""
        if len(quotes) < lookback + 1 or len(indicator_values) < lookback + 1:
            return {'bullish_divergence': False, 'bearish_divergence': False}
        
        closes = [q['close'] for q in quotes]
        
        # Find price peaks and troughs
        price_peaks = []
        price_troughs = []
        
        for i in range(2, len(closes) - 2):
            # Peak
            if closes[i] > closes[i-1] and closes[i] > closes[i+1] and \
               closes[i] > closes[i-2] and closes[i] > closes[i+2]:
                price_peaks.append((i, closes[i]))
            
            # Trough
            if closes[i] < closes[i-1] and closes[i] < closes[i+1] and \
               closes[i] < closes[i-2] and closes[i] < closes[i+2]:
                price_troughs.append((i, closes[i]))
        
        # Find indicator peaks and troughs
        indicator_peaks = []
        indicator_troughs = []
        
        for i in range(2, len(indicator_values) - 2):
            # Peak
            if indicator_values[i] > indicator_values[i-1] and indicator_values[i] > indicator_values[i+1] and \
               indicator_values[i] > indicator_values[i-2] and indicator_values[i] > indicator_values[i+2]:
                indicator_peaks.append((i, indicator_values[i]))
            
            # Trough
            if indicator_values[i] < indicator_values[i-1] and indicator_values[i] < indicator_values[i+1] and \
               indicator_values[i] < indicator_values[i-2] and indicator_values[i] < indicator_values[i+2]:
                indicator_troughs.append((i, indicator_values[i]))
        
        # Detect divergences
        bullish_divergence = False
        bearish_divergence = False
        
        # Bullish divergence: price makes lower lows, indicator makes higher lows
        if len(price_troughs) >= 2 and len(indicator_troughs) >= 2:
            recent_price_trough = price_troughs[-1]
            previous_price_trough = price_troughs[-2]
            
            recent_indicator_trough = None
            previous_indicator_trough = None
            
            for trough in indicator_troughs:
                if trough[0] <= recent_price_trough[0]:
                    recent_indicator_trough = trough
                    break
            
            for trough in reversed(indicator_troughs):
                if trough[0] <= previous_price_trough[0] and trough != recent_indicator_trough:
                    previous_indicator_trough = trough
                    break
            
            if recent_price_trough and previous_price_trough and recent_indicator_trough and previous_indicator_trough:
                if (recent_price_trough[1] < previous_price_trough[1] and 
                    recent_indicator_trough[1] > previous_indicator_trough[1]):
                    bullish_divergence = True
        
        # Bearish divergence: price makes higher highs, indicator makes lower highs
        if len(price_peaks) >= 2 and len(indicator_peaks) >= 2:
            recent_price_peak = price_peaks[-1]
            previous_price_peak = price_peaks[-2]
            
            recent_indicator_peak = None
            previous_indicator_peak = None
            
            for peak in indicator_peaks:
                if peak[0] <= recent_price_peak[0]:
                    recent_indicator_peak = peak
                    break
            
            for peak in reversed(indicator_peaks):
                if peak[0] <= previous_price_peak[0] and peak != recent_indicator_peak:
                    previous_indicator_peak = peak
                    break
            
            if recent_price_peak and previous_price_peak and recent_indicator_peak and previous_indicator_peak:
                if (recent_price_peak[1] > previous_price_peak[1] and 
                    recent_indicator_peak[1] < previous_indicator_peak[1]):
                    bearish_divergence = True
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }
    
    def detect_candlestick_patterns(self, quotes: List[Dict]) -> Dict[str, bool]:
        """Detect basic candlestick patterns"""
        if len(quotes) < 3:
            return {
                'bullish_engulfing': False, 'bearish_engulfing': False,
                'hammer': False, 'shooting_star': False, 'doji': False
            }
        
        current = quotes[-1]
        previous = quotes[-2]
        
        # Current candle components
        current_body = abs(current['close'] - current['open'])
        current_range = current['high'] - current['low']
        current_upper_shadow = current['high'] - max(current['open'], current['close'])
        current_lower_shadow = min(current['open'], current['close']) - current['low']
        
        # Previous candle components
        previous_body = abs(previous['close'] - previous['open'])
        previous_range = previous['high'] - previous['low']
        
        patterns = {
            'bullish_engulfing': False,
            'bearish_engulfing': False,
            'hammer': False,
            'shooting_star': False,
            'doji': False
        }
        
        # Doji (body less than 10% of range)
        if current_body < current_range * 0.1:
            patterns['doji'] = True
        
        # Hammer (small body, long lower shadow, short upper shadow)
        if (current_lower_shadow > current_body * 2 and 
            current_upper_shadow < current_body * 0.5 and
            current_body < current_range * 0.3):
            patterns['hammer'] = True
        
        # Shooting Star (small body, long upper shadow, short lower shadow)
        if (current_upper_shadow > current_body * 2 and 
            current_lower_shadow < current_body * 0.5 and
            current_body < current_range * 0.3):
            patterns['shooting_star'] = True
        
        # Bullish Engulfing (current green candle engulfs previous red candle)
        if (current['close'] > current['open'] and 
            previous['close'] < previous['open'] and
            current['open'] < previous['close'] and
            current['close'] > previous['open']):
            patterns['bullish_engulfing'] = True
        
        # Bearish Engulfing (current red candle engulfs previous green candle)
        if (current['close'] < current['open'] and 
            previous['close'] > previous['open'] and
            current['open'] > previous['close'] and
            current['close'] < previous['open']):
            patterns['bearish_engulfing'] = True
        
        return patterns
    
    def _calculate_ema(self, values: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(values) < period:
            return sum(values) / len(values) if values else 0
        
        multiplier = 2.0 / (period + 1)
        ema = sum(values[:period]) / period
        
        for i in range(period, len(values)):
            ema = (values[i] - ema) * multiplier + ema
        
        return ema
    
    def market_regime(self, quotes: List[Dict], period: int = 20) -> Dict[str, str]:
        """Determine market regime: trend, range, or high volatility"""
        if len(quotes) < period:
            return {'regime': 'unknown', 'confidence': 0.0}
        
        # Calculate various indicators
        adx_data = self.adx_dmi(quotes, period)
        bb_data = self.bollinger_bands(quotes, period)
        atr_data = self.atr(quotes, period)
        
        closes = [q['close'] for q in quotes]
        
        # Calculate volatility
        returns = []
        for i in range(1, len(closes)):
            returns.append(math.log(closes[i] / closes[i-1]))
        
        volatility = math.sqrt(sum(r**2 for r in returns[-period:]) / period) if returns else 0
        
        # Regime classification
        regime = 'range'
        confidence = 0.5
        
        # Strong trend criteria
        if adx_data['adx'] > 25 and abs(adx_data['plus_di'] - adx_data['minus_di']) > 15:
            regime = 'trend'
            confidence = min(adx_data['adx'] / 50, 1.0)
        
        # High volatility criteria
        elif volatility > 0.02:  # 2% daily volatility threshold
            regime = 'high_volatility'
            confidence = min(volatility / 0.05, 1.0)
        
        # Range criteria
        elif bb_data['width'] / closes[-1] < 0.02:  # Narrow bands
            regime = 'range'
            confidence = 0.7
        
        return {
            'regime': regime,
            'confidence': confidence,
            'adx': adx_data['adx'],
            'volatility': volatility,
            'bb_width_ratio': bb_data['width'] / closes[-1] if closes else 0
        }
    
    def bollinger_bands(self, quotes: List[Dict], period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        if len(quotes) < period:
            return {'upper': 0, 'middle': 0, 'lower': 0, 'width': 0, 'position': 0}
        
        closes = [q['close'] for q in quotes]
        
        middle_band = sum(closes[-period:]) / period
        variance = sum((close - middle_band) ** 2 for close in closes[-period:]) / period
        std = math.sqrt(variance)
        
        upper_band = middle_band + std_dev * std
        lower_band = middle_band - std_dev * std
        width = upper_band - lower_band
        
        # Position in bands (0-1, where 0.5 is middle)
        position = (closes[-1] - lower_band) / width if width > 0 else 0.5
        position = max(0, min(1, position))
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band,
            'width': width,
            'position': position
        }
    
    def atr(self, quotes: List[Dict], period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(quotes) < 2:
            return 0
        
        highs = [q['high'] for q in quotes]
        lows = [q['low'] for q in quotes]
        closes = [q['close'] for q in quotes]
        
        tr_values = []
        for i in range(1, len(quotes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_values.append(tr)
        
        if len(tr_values) < period:
            return sum(tr_values) / len(tr_values) if tr_values else 0
        
        atr = sum(tr_values[:period]) / period
        for i in range(period, len(tr_values)):
            atr = (atr * (period - 1) + tr_values[i]) / period
        
        return atr
    
    def rsi(self, quotes: List[Dict], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(quotes) < period + 1:
            return 50
        
        closes = [q['close'] for q in quotes]
        
        gains = []
        losses = []
        
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def macd(self, quotes: List[Dict], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, float]:
        """Calculate MACD indicator"""
        if len(quotes) < slow_period + signal_period:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        closes = [q['close'] for q in quotes]
        
        # Calculate EMAs
        fast_ema = self._calculate_ema(closes, fast_period)
        slow_ema = self._calculate_ema(closes, slow_period)
        
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD)
        macd_values = []
        for i in range(slow_period, len(closes)):
            fast = self._calculate_ema(closes[:i+1], fast_period)
            slow = self._calculate_ema(closes[:i+1], slow_period)
            macd_values.append(fast - slow)
        
        signal_line = self._calculate_ema(macd_values, signal_period) if macd_values else 0
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }