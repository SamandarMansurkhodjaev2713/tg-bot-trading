import numpy as np
import sqlite3
import json
import math
import os
import pandas as pd
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import pickle
import os

from forex_indicators import AdvancedTechnicalIndicators
from data_sources import DataManager

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

class AdvancedTradingAI:
    """Advanced AI trading system with ensemble models, calibration, and meta-labeling"""
    
    def __init__(self, db_path: str = "ai_trading_signals.db"):
        self.db_path = db_path
        self.indicators = AdvancedTechnicalIndicators()
        self.data_manager = DataManager({})
        
        # Model components
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        
        # Calibration
        self.calibrators = {}
        
        # Meta-labeling components
        self.meta_model = None
        self.meta_scaler = None
        
        # Performance tracking
        self.performance_metrics = {}
        
        # Market regime classifiers
        self.regime_model = None
        self.regime_scaler = None
        
        # Initialize database
        self.init_database()
        
        # Load models if they exist
        self.load_models()

    def build_unified_dataset(self, pairs: List[str], timeframe: str, limit: int = 1000) -> Dict[str, List[Dict]]:
        result = {}
        for p in pairs:
            data = asyncio.run(self.data_manager.get_merged_market_data(p, timeframe, limit))
            result[p] = self._clean_and_align(data, timeframe)
        return result

    def _clean_and_align(self, quotes: List[Dict], timeframe: str) -> List[Dict]:
        if not quotes:
            return []
        df = pd.DataFrame(quotes)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.dropna()
        df = df.sort_values('timestamp')
        df = df.set_index('timestamp')
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
        rule = {'1m':'1min','5m':'5min','15m':'15min','30m':'30min','1h':'1h','4h':'4h','1d':'1d'}.get(timeframe,'1h')
        df = df.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        r = df['close'].pct_change().fillna(0)
        mad = np.median(np.abs(r - np.median(r))) if len(r) else 0
        if mad > 0:
            thr = 10 * mad
            r = np.clip(r, -thr, thr)
            df['close'] = df['close'].shift(1) * (1 + r)
        df = df.dropna()
        df['pair'] = quotes[0]['pair']
        df['timeframe'] = timeframe
        out = []
        for idx, row in df.iterrows():
            out.append({
                'timestamp': idx.isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row.get('volume', 0.0)),
                'pair': row['pair'],
                'timeframe': timeframe
            })
        return out

    def extract_multi_timeframe_features(self, quotes: List[Dict], pair: str) -> Dict[str, float]:
        feats = {}
        base = self.extract_advanced_features(quotes, pair)
        feats.update({f"{k}": v for k, v in base.items()})
        df = pd.DataFrame(quotes)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        for tf, rule in [('15m','15min'),('1h','1h'),('4h','4h')]:
            rdf = df.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            qs = []
            for idx, row in rdf.tail(120).iterrows():
                qs.append({'timestamp': idx.isoformat(), 'open': float(row['open']), 'high': float(row['high']), 'low': float(row['low']), 'close': float(row['close']), 'volume': float(row.get('volume',0))})
            f = self.extract_advanced_features(qs, pair)
            for k, v in f.items():
                feats[f"{tf}_{k}"] = v
        return feats

    def extract_correlation_features(self, main_pair: str, main_quotes: List[Dict], other_data: Dict[str, List[Dict]]) -> Dict[str, float]:
        feats = {}
        mr = pd.Series([q['close'] for q in main_quotes]).pct_change().dropna()
        for op, oq in other_data.items():
            if not oq:
                continue
            orr = pd.Series([q['close'] for q in oq]).pct_change().dropna()
            n = min(len(mr), len(orr))
            if n < 20:
                continue
            c = float(pd.Series(mr.tail(n)).corr(pd.Series(orr.tail(n))))
            feats[f"corr_{op}"] = c
        return feats

    def balance_training(self, X: List[Dict[str, float]], y: List[int]) -> Tuple[List[Dict[str,float]], List[int]]:
        pos = [(xi, yi) for xi, yi in zip(X, y) if yi == 1]
        neg = [(xi, yi) for xi, yi in zip(X, y) if yi == -1]
        neu = [(xi, yi) for xi, yi in zip(X, y) if yi == 0]
        m = min(len(pos), len(neg), max(1, len(neu)))
        balanced = pos[:m] + neg[:m] + neu[:m]
        Xb = [b[0] for b in balanced]
        yb = [b[1] for b in balanced]
        return Xb, yb

    def train_improved(self, data: Dict[str, List[Dict]], timeframe: str = '1h') -> Dict[str, Any]:
        features = []
        labels = []
        for pair, quotes in data.items():
            if len(quotes) < 150:
                continue
            closes = pd.Series([q['close'] for q in quotes])
            for i in range(120, len(quotes)-1):
                window = quotes[:i+1]
                f = self.extract_multi_timeframe_features(window, pair)
                features.append(f)
                ch = float(closes.pct_change().iloc[i+1])
                if ch > 0.0002:
                    labels.append(1)
                elif ch < -0.0002:
                    labels.append(-1)
                else:
                    labels.append(0)
        if not features:
            return {'trained': False, 'samples': 0, 'tcn_used': False}
        Xb, yb = self.balance_training(features, labels)
        self._train_models(Xb, yb)
        tcn_used = self.train_tcn_optional(Xb, yb)
        return {'trained': True, 'samples': len(Xb), 'tcn_used': tcn_used}

    def predict_enhanced(self, quotes: List[Dict], pair: str) -> Dict[str, Any]:
        f = self.extract_multi_timeframe_features(quotes, pair)
        s, p, c = self.predict(f, use_calibration=False)
        vol = float(np.std(pd.Series([q['close'] for q in quotes]).pct_change().dropna().tail(50))) if len(quotes) > 50 else 0.0
        return {'direction': 'up' if s == 1 else 'down' if s == -1 else 'flat', 'probability': p, 'confidence': c, 'volatility': vol}

    def train_tcn_optional(self, features_list: List[Dict[str,float]], labels: List[int]) -> bool:
        if torch is None:
            return False
        fn = sorted(list({k for f in features_list for k in f.keys()}))
        X = np.array([[f.get(k,0.0) for k in fn] for f in features_list], dtype=np.float32)
        y = np.array([1 if yy==1 else 0 for yy in labels], dtype=np.int64)
        X = torch.tensor(X).unsqueeze(1)
        y = torch.tensor(y)
        class SimpleTCN(nn.Module):
            def __init__(self, n_features):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
                self.fc = nn.Linear(n_features*16, 2)
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = x.view(x.size(0), -1)
                return self.fc(x)
        model = SimpleTCN(X.shape[-1])
        opt = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(5):
            opt.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
        return True

    def run_full_pipeline(self, pairs: List[str], timeframe: str = '1h', limit: int = 1000) -> Dict[str, Any]:
        data = self.build_unified_dataset(pairs, timeframe, limit)
        train_res = self.train_improved(data, timeframe)
        preds = {}
        for p in pairs:
            q = data.get(p, [])
            if q:
                preds[p] = self.predict_enhanced(q, p)
        # Baseline vs Improved CV
        base_feats, base_labels = self._build_samples(data, use_mtf=False)
        mtf_feats, mtf_labels = self._build_samples(data, use_mtf=True)
        baseline = self._time_series_cv_metrics(base_feats, base_labels)
        improved = self._time_series_cv_metrics(mtf_feats, mtf_labels)
        return {'train': train_res, 'predictions': preds, 'baseline_cv': baseline, 'improved_cv': improved}

    def _build_samples(self, data: Dict[str, List[Dict]], use_mtf: bool) -> Tuple[List[Dict[str,float]], List[int]]:
        features = []
        labels = []
        for pair, quotes in data.items():
            if len(quotes) < 160:
                continue
            closes = pd.Series([q['close'] for q in quotes])
            for i in range(120, len(quotes)-1):
                window = quotes[:i+1]
                f = self.extract_multi_timeframe_features(window, pair) if use_mtf else self.extract_advanced_features(window, pair)
                features.append(f)
                ch = float(closes.pct_change().iloc[i+1])
                if ch > 0.0002:
                    labels.append(1)
                elif ch < -0.0002:
                    labels.append(-1)
                else:
                    labels.append(0)
        return features, labels

    def _time_series_cv_metrics(self, features_list: List[Dict[str,float]], labels: List[int]) -> Dict[str, float]:
        if not features_list or len(features_list) < 50:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        X, feature_names = self._prepare_training_data(features_list)
        y = np.array(labels)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        tscv = TimeSeriesSplit(n_splits=3)
        accs = []
        precs = []
        recs = []
        f1s = []
        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        for tr, te in tscv.split(Xs):
            clf.fit(Xs[tr], y[tr])
            pred = clf.predict(Xs[te])
            accs.append(accuracy_score(y[te], pred))
            precs.append(precision_score(y[te], pred, average='weighted', zero_division=0))
            recs.append(recall_score(y[te], pred, average='weighted', zero_division=0))
            f1s.append(f1_score(y[te], pred, average='weighted', zero_division=0))
        return {
            'accuracy': float(np.mean(accs)),
            'precision': float(np.mean(precs)),
            'recall': float(np.mean(recs)),
            'f1': float(np.mean(f1s))
        }
    
    def init_database(self):
        """Initialize SQLite database for signal tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                probability REAL NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                expected_value REAL,
                market_regime TEXT,
                features TEXT,
                outcome TEXT,
                pnl REAL,
                metadata TEXT
            )
        ''')
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                pair TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                position_size REAL,
                pnl REAL,
                outcome TEXT,
                duration_minutes INTEGER,
                metadata TEXT,
                FOREIGN KEY (signal_id) REFERENCES ai_signals (id)
            )
        ''')
        
        # Model performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                period_days INTEGER,
                market_regime TEXT,
                metadata TEXT
            )
        ''')
        
        # Market data cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL,
                metadata TEXT,
                UNIQUE(pair, timeframe, timestamp)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_advanced_features(self, quotes: List[Dict], pair: str = "") -> Dict[str, float]:
        """Extract 48+ advanced technical features"""
        if len(quotes) < 52:  # Minimum for Ichimoku
            return {}
        
        features = {}
        
        # Price-based features
        closes = [q['close'] for q in quotes]
        current_price = closes[-1]
        
        # Moving averages and slopes
        for period in [5, 10, 20, 50]:
            if len(quotes) >= period:
                ma = sum(closes[-period:]) / period
                features[f'sma_{period}'] = ma
                features[f'sma_{period}_slope'] = (ma - sum(closes[-period-1:-1]) / period) if len(closes) > period else 0
                features[f'price_vs_sma_{period}'] = (current_price - ma) / ma if ma != 0 else 0
        
        # EMA slopes
        for period in [8, 21, 34]:
            if len(quotes) >= period:
                ema = self._calculate_ema(closes, period)
                features[f'ema_{period}'] = ema
                prev_ema = self._calculate_ema(closes[:-1], period) if len(closes) > 1 else ema
                features[f'ema_{period}_slope'] = (ema - prev_ema) / prev_ema if prev_ema != 0 else 0
        
        # RSI features
        rsi_14 = self.indicators.rsi(quotes, 14)
        features['rsi_14'] = rsi_14
        features['rsi_14_zscore'] = (rsi_14 - 50) / 25  # Normalize to [-1, 1]
        
        # RSI divergences
        rsi_values = []
        for i in range(20, len(quotes)):
            rsi_values.append(self.indicators.rsi(quotes[:i+1], 14))
        
        if len(rsi_values) >= 10:
            divergences = self.indicators.detect_divergences(quotes[-20:], rsi_values[-20:])
            features['rsi_bullish_divergence'] = 1.0 if divergences['bullish_divergence'] else 0.0
            features['rsi_bearish_divergence'] = 1.0 if divergences['bearish_divergence'] else 0.0
        
        # MACD features
        macd_data = self.indicators.macd(quotes)
        features['macd'] = macd_data['macd']
        features['macd_signal'] = macd_data['signal']
        features['macd_histogram'] = macd_data['histogram']
        features['macd_above_signal'] = 1.0 if macd_data['macd'] > macd_data['signal'] else 0.0
        
        # Bollinger Bands features
        bb_data = self.indicators.bollinger_bands(quotes)
        features['bb_position'] = bb_data['position']
        features['bb_width'] = bb_data['width']
        features['bb_width_normalized'] = bb_data['width'] / current_price if current_price != 0 else 0
        
        # ATR and volatility
        atr_14 = self.indicators.atr(quotes, 14)
        features['atr_14'] = atr_14
        features['atr_14_normalized'] = atr_14 / current_price if current_price != 0 else 0
        
        # Volatility features
        returns = []
        for i in range(1, len(closes)):
            returns.append(math.log(closes[i] / closes[i-1]))
        
        if returns:
            features['volatility_20'] = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
            features['volatility_5'] = np.std(returns[-5:]) if len(returns) >= 5 else np.std(returns)
            features['volatility_ratio'] = features['volatility_5'] / features['volatility_20'] if features['volatility_20'] > 0 else 1.0
        
        # ADX/DMI features
        adx_data = self.indicators.adx_dmi(quotes)
        features['adx'] = adx_data['adx']
        features['plus_di'] = adx_data['plus_di']
        features['minus_di'] = adx_data['minus_di']
        features['adx_trend_strength'] = adx_data['trend_strength']
        features['dmi_spread'] = abs(adx_data['plus_di'] - adx_data['minus_di'])
        
        # Aroon features
        aroon_data = self.indicators.aroon(quotes)
        features['aroon_up'] = aroon_data['aroon_up']
        features['aroon_down'] = aroon_data['aroon_down']
        features['aroon_oscillator'] = aroon_data['aroon_oscillator']
        
        # Supertrend
        supertrend_data = self.indicators.supertrend(quotes)
        features['supertrend'] = supertrend_data['supertrend']
        features['supertrend_direction'] = supertrend_data['direction']
        features['supertrend_distance'] = (current_price - supertrend_data['supertrend']) / current_price if current_price != 0 else 0
        
        # Keltner Channels
        keltner_data = self.indicators.keltner_channels(quotes)
        features['keltner_position'] = (current_price - keltner_data['lower']) / keltner_data['width'] if keltner_data['width'] > 0 else 0.5
        
        # Donchian Channels
        donchian_data = self.indicators.donchian_channels(quotes)
        features['donchian_position'] = (current_price - donchian_data['lower']) / donchian_data['width'] if donchian_data['width'] > 0 else 0.5
        
        # Hurst exponent
        hurst = self.indicators.hurst_exponent(quotes)
        features['hurst_exponent'] = hurst
        features['market_memory'] = hurst  # Same as Hurst, different name for clarity
        
        # Stochastic
        stochastic_data = self.indicators.stochastic(quotes)
        features['stochastic_k'] = stochastic_data['k']
        features['stochastic_d'] = stochastic_data['d']
        features['stochastic_position'] = stochastic_data['k'] / 100.0
        
        # CCI
        cci_20 = self.indicators.cci(quotes, 20)
        features['cci_20'] = cci_20
        features['cci_20_zscore'] = cci_20 / 100.0  # Normalize roughly to [-1, 1]
        
        # MFI
        mfi_14 = self.indicators.mfi(quotes)
        features['mfi_14'] = mfi_14
        features['mfi_14_zscore'] = (mfi_14 - 50) / 25  # Normalize to [-1, 1]
        
        # Heikin Ashi features
        ha_data = self.indicators.heikin_ashi(quotes)
        if len(ha_data['close']) >= 3:
            # Bullish HA pattern: current HA close > HA open and previous HA close > HA open
            current_ha_bullish = ha_data['close'][-1] > ha_data['open'][-1]
            prev_ha_bullish = ha_data['close'][-2] > ha_data['open'][-2]
            features['ha_trend_bullish'] = 1.0 if current_ha_bullish and prev_ha_bullish else 0.0
            
            # Bearish HA pattern: current HA close < HA open and previous HA close < HA open
            current_ha_bearish = ha_data['close'][-1] < ha_data['open'][-1]
            prev_ha_bearish = ha_data['close'][-2] < ha_data['open'][-2]
            features['ha_trend_bearish'] = 1.0 if current_ha_bearish and prev_ha_bearish else 0.0
            
            # HA trend strength
            features['ha_trend_strength'] = abs(ha_data['close'][-1] - ha_data['open'][-1]) / ha_data['open'][-1] if ha_data['open'][-1] != 0 else 0
        
        # VWAP features
        vwap_values = self.indicators.vwap(quotes)
        if vwap_values:
            current_vwap = vwap_values[-1]
            features['vwap'] = current_vwap
            features['price_vs_vwap'] = (current_price - current_vwap) / current_vwap if current_vwap != 0 else 0
            
            # VWAP deviation bands (1 standard deviation)
            if len(vwap_values) >= 20:
                vwap_std = np.std(vwap_values[-20:])
                features['vwap_upper_1std'] = current_vwap + vwap_std
                features['vwap_lower_1std'] = current_vwap - vwap_std
                features['vwap_position'] = (current_price - (current_vwap - vwap_std)) / (2 * vwap_std) if vwap_std > 0 else 0.5
        features['mfi_position'] = mfi_14 / 100.0
        
        # OBV
        obv = self.indicators.obv(quotes)
        features['obv'] = obv
        if len(quotes) > 1:
            prev_obv = self.indicators.obv(quotes[:-1])
            features['obv_slope'] = (obv - prev_obv) / abs(prev_obv) if prev_obv != 0 else 0
        
        # Ichimoku
        ichimoku_data = self.indicators.ichimoku(quotes)
        features['tenkan'] = ichimoku_data['tenkan']
        features['kijun'] = ichimoku_data['kijun']
        features['senkou_a'] = ichimoku_data['senkou_a']
        features['senkou_b'] = ichimoku_data['senkou_b']
        features['chikou'] = ichimoku_data['chikou']
        features['cloud_top'] = ichimoku_data['cloud_top']
        features['cloud_bottom'] = ichimoku_data['cloud_bottom']
        features['price_vs_cloud'] = (current_price - ichimoku_data['cloud_bottom']) / (ichimoku_data['cloud_top'] - ichimoku_data['cloud_bottom']) if (ichimoku_data['cloud_top'] - ichimoku_data['cloud_bottom']) > 0 else 0.5
        
        # Market regime
        regime_data = self.indicators.market_regime(quotes)
        features['market_regime'] = 1.0 if regime_data['regime'] == 'trend' else 0.0
        features['regime_confidence'] = regime_data['confidence']
        features['regime_adx'] = regime_data['adx']
        features['regime_volatility'] = regime_data['volatility']
        
        # Candlestick patterns
        patterns = self.indicators.detect_candlestick_patterns(quotes[-10:])
        features['bullish_engulfing'] = 1.0 if patterns['bullish_engulfing'] else 0.0
        features['bearish_engulfing'] = 1.0 if patterns['bearish_engulfing'] else 0.0
        features['hammer'] = 1.0 if patterns['hammer'] else 0.0
        features['shooting_star'] = 1.0 if patterns['shooting_star'] else 0.0
        features['doji'] = 1.0 if patterns['doji'] else 0.0
        
        # Cross-asset features (if pair is provided)
        if pair:
            features.update(self._get_cross_asset_features(pair, quotes))
        
        # Time-based features
        if quotes and 'timestamp' in quotes[-1]:
            dt = datetime.fromisoformat(quotes[-1]['timestamp'])
            features['hour_of_day'] = dt.hour / 24.0
            features['day_of_week'] = dt.weekday() / 7.0
            features['month'] = dt.month / 12.0
        
        # Z-score normalizations
        zscore_features = {}
        for key, value in features.items():
            if isinstance(value, (int, float)) and not key.endswith('_zscore'):
                zscore_features[f'{key}_zscore'] = self._calculate_zscore(value, key)
        features.update(zscore_features)
        
        return features
    
    def _get_cross_asset_features(self, pair: str, quotes: List[Dict]) -> Dict[str, float]:
        """Get cross-asset features for major pairs"""
        features = {}
        
        # Map major pairs to their drivers
        cross_assets = {
            'EURUSD': ['DXY', 'US10Y'],
            'GBPUSD': ['DXY', 'US10Y'],
            'USDJPY': ['DXY', 'US10Y'],
            'XAUUSD': ['DXY', 'US10Y', 'VIX'],
            'XAGUSD': ['DXY', 'US10Y', 'VIX'],
            'US30': ['DXY', 'US10Y', 'VIX'],
            'US100': ['DXY', 'US10Y', 'VIX'],
            'US500': ['DXY', 'US10Y', 'VIX'],
            'USTEC': ['DXY', 'US10Y', 'VIX'],
            'USOIL': ['DXY', 'VIX'],
            'BTCUSD': ['DXY', 'VIX'],
            'ETHUSD': ['DXY', 'VIX']
        }
        
        if pair in cross_assets:
            # These would normally be fetched from external sources
            # For now, we'll use placeholder calculations
            current_price = quotes[-1]['close']
            
            # Simulate DXY correlation (inverse for most pairs)
            if pair in ['EURUSD', 'GBPUSD', 'XAUUSD', 'XAGUSD', 'BTCUSD', 'ETHUSD']:
                features['dxy_correlation'] = -0.7 + np.random.normal(0, 0.1)
            else:
                features['dxy_correlation'] = 0.3 + np.random.normal(0, 0.1)
            
            # Simulate VIX correlation (risk-off pairs)
            if pair in ['XAUUSD', 'XAGUSD', 'US30', 'US100', 'US500', 'USTEC', 'BTCUSD', 'ETHUSD']:
                features['vix_correlation'] = 0.5 + np.random.normal(0, 0.1)
            else:
                features['vix_correlation'] = -0.2 + np.random.normal(0, 0.1)
            
            # Simulate US10Y correlation
            if pair in ['USDJPY', 'US30', 'US100', 'US500', 'USTEC']:
                features['us10y_correlation'] = 0.6 + np.random.normal(0, 0.1)
            else:
                features['us10y_correlation'] = -0.3 + np.random.normal(0, 0.1)
        
        return features
    
    def _calculate_zscore(self, value: float, feature_name: str) -> float:
        """Calculate z-score for a feature (placeholder - would use historical stats)"""
        # This would normally use historical mean and std for the feature
        # For now, return normalized value
        if 'rsi' in feature_name:
            return (value - 50) / 25  # RSI: 0-100 -> [-2, 2]
        elif 'stochastic' in feature_name:
            return (value - 50) / 25  # Stochastic: 0-100 -> [-2, 2]
        elif 'adx' in feature_name:
            return (value - 25) / 15  # ADX: 0-100 -> [-1.67, 5]
        elif 'macd' in feature_name:
            return max(-2, min(2, value / 100))  # Rough normalization
        else:
            return max(-2, min(2, value))  # Default clamp
    
    def _calculate_ema(self, values: List[float], period: int) -> float:
        """Calculate EMA"""
        if len(values) < period:
            return sum(values) / len(values) if values else 0
        
        multiplier = 2.0 / (period + 1)
        ema = sum(values[:period]) / period
        
        for i in range(period, len(values)):
            ema = (values[i] - ema) * multiplier + ema
        
        return ema
    
    def triple_barrier_labeling(self, quotes: List[Dict], upper_barrier: float, lower_barrier: float, 
                               time_barrier: int = 20) -> Dict[str, Any]:
        """Implement triple-barrier labeling (Lopez de Prado method)"""
        if len(quotes) < time_barrier + 1:
            return {'label': 0, 'touch_time': None, 'barrier_touched': None}
        
        current_price = quotes[-1]['close']
        
        # Future prices
        future_quotes = quotes[-time_barrier:]
        
        # Check which barrier is touched first
        upper_touch_time = None
        lower_touch_time = None
        
        for i, quote in enumerate(future_quotes):
            if quote['high'] >= current_price * (1 + upper_barrier):
                upper_touch_time = i
                break
        
        for i, quote in enumerate(future_quotes):
            if quote['low'] <= current_price * (1 - lower_barrier):
                lower_touch_time = i
                break
        
        # Determine outcome
        if upper_touch_time is not None and lower_touch_time is not None:
            if upper_touch_time < lower_touch_time:
                label = 1  # Upper barrier touched first (profit)
                touch_time = upper_touch_time
                barrier_touched = 'upper'
            else:
                label = -1  # Lower barrier touched first (loss)
                touch_time = lower_touch_time
                barrier_touched = 'lower'
        elif upper_touch_time is not None:
            label = 1
            touch_time = upper_touch_time
            barrier_touched = 'upper'
        elif lower_touch_time is not None:
            label = -1
            touch_time = lower_touch_time
            barrier_touched = 'lower'
        else:
            label = 0  # Time barrier reached first (neutral)
            touch_time = time_barrier
            barrier_touched = 'time'
        
        return {
            'label': label,
            'touch_time': touch_time,
            'barrier_touched': barrier_touched,
            'upper_barrier': upper_barrier,
            'lower_barrier': lower_barrier,
            'time_barrier': time_barrier
        }
    
    def meta_labeling(self, primary_signal: int, features: Dict[str, float], 
                     meta_threshold: float = 0.55) -> Tuple[int, float]:
        """Apply meta-labeling to filter primary signals"""
        if self.meta_model is None:
            return primary_signal, 0.5
        
        # Prepare features for meta-model
        meta_features = self._prepare_meta_features(features)
        
        # Get meta-model prediction
        if hasattr(self.meta_model, 'predict_proba'):
            meta_proba = self.meta_model.predict_proba([meta_features])[0]
            meta_confidence = meta_proba[1] if len(meta_proba) > 1 else meta_proba[0]
        else:
            meta_prediction = self.meta_model.predict([meta_features])[0]
            meta_confidence = 0.6 if meta_prediction == 1 else 0.4
        
        # Apply meta-labeling filter
        if primary_signal != 0 and meta_confidence >= meta_threshold:
            return primary_signal, meta_confidence
        else:
            return 0, meta_confidence
    
    def _prepare_meta_features(self, features: Dict[str, float]) -> List[float]:
        """Prepare features for meta-model"""
        # Select most important features for meta-labeling
        important_features = [
            'adx', 'rsi_14', 'macd_histogram', 'bb_position', 'atr_14_normalized',
            'market_regime', 'regime_confidence', 'volatility_20', 'hurst_exponent',
            'stochastic_k', 'cci_20', 'mfi_14', 'aroon_oscillator'
        ]
        
        meta_features = []
        for feature in important_features:
            meta_features.append(features.get(feature, 0.0))
        
        return meta_features
    
    def walk_forward_training(self, quotes_list: List[List[Dict]], labels: List[int], 
                            features_list: List[Dict[str, float]], 
                            window_size: int = 100, step_size: int = 20) -> Dict[str, float]:
        """Implement walk-forward training with purged/embargoed cross-validation"""
        if len(quotes_list) < window_size + step_size:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        results = []
        
        # Time series split with embargo
        for start_idx in range(0, len(quotes_list) - window_size, step_size):
            train_end = start_idx + window_size
            test_start = train_end + 5  # 5-sample embargo
            test_end = min(test_start + step_size, len(quotes_list))
            
            if test_end <= test_start:
                continue
            
            # Training data
            train_features = features_list[start_idx:train_end]
            train_labels = labels[start_idx:train_end]
            
            # Test data (with embargo)
            test_features = features_list[test_start:test_end]
            test_labels = labels[test_start:test_end]
            
            # Train models
            self._train_models(train_features, train_labels)
            
            # Evaluate on test set
            predictions = self.predict_batch(test_features)
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, predictions)
            precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(test_labels, predictions, average='weighted', zero_division=0)
            f1 = f1_score(test_labels, predictions, average='weighted', zero_division=0)
            
            results.append({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'window': start_idx
            })
        
        # Average results
        if results:
            avg_results = {
                'accuracy': np.mean([r['accuracy'] for r in results]),
                'precision': np.mean([r['precision'] for r in results]),
                'recall': np.mean([r['recall'] for r in results]),
                'f1': np.mean([r['f1'] for r in results])
            }
        else:
            avg_results = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        return avg_results
    
    def _train_models(self, features_list: List[Dict[str, float]], labels: List[int]):
        """Train ensemble models"""
        # Prepare data
        X, feature_names = self._prepare_training_data(features_list)
        y = np.array(labels)
        
        if len(X) < 10 or len(np.unique(y)) < 2:
            return
        
        # Initialize scalers
        self.scalers['main'] = StandardScaler()
        X_scaled = self.scalers['main'].fit_transform(X)
        
        # Train ensemble models
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        )
        self.models['gradient_boost'] = GradientBoostingClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42
        )
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
        )
        
        # Train models
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y)
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        # Train calibration models
        self._train_calibration_models(X_scaled, y)
        
        # Train meta-model
        self._train_meta_model(X_scaled, y)
        
        # Store feature names
        self.feature_names = feature_names
    
    def _prepare_training_data(self, features_list: List[Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
        """Prepare training data from feature dictionaries"""
        if not features_list:
            return np.array([]), []
        
        # Get all feature names
        all_features = set()
        for features in features_list:
            all_features.update(features.keys())
        
        feature_names = sorted(list(all_features))
        
        # Create feature matrix
        X = []
        for features in features_list:
            row = []
            for feature_name in feature_names:
                row.append(features.get(feature_name, 0.0))
            X.append(row)
        
        return np.array(X), feature_names
    
    def _train_calibration_models(self, X: np.ndarray, y: np.ndarray):
        """Train probability calibration models"""
        try:
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    if proba.ndim == 2 and proba.shape[1] > 1:
                        p1 = proba[:, 1] if model.classes_[1] == 1 else proba[:, 0]
                    else:
                        p1 = proba.ravel()
                    iso_calibrator = IsotonicRegression(out_of_bounds='clip')
                    iso_calibrator.fit(p1, y)
                    self.calibrators[f'{name}_isotonic'] = iso_calibrator
        except Exception as e:
            print(f"Error in calibration training: {e}")
    
    def _train_meta_model(self, X: np.ndarray, y: np.ndarray):
        """Train meta-model for meta-labeling"""
        try:
            # Use logistic regression for meta-labeling
            self.meta_model = LogisticRegression(random_state=42)
            
            # Prepare meta-features (simplified feature set)
            meta_features = []
            for i in range(len(X)):
                features_dict = dict(zip(self.feature_names, X[i]))
                meta_features.append(self._prepare_meta_features(features_dict))
            
            self.meta_model.fit(meta_features, y)
            self.meta_scaler = None  # No scaling needed for meta-model
            
        except Exception as e:
            print(f"Error training meta-model: {e}")
    
    def predict(self, features: Dict[str, float], use_calibration: bool = True) -> Tuple[int, float, str]:
        """Make prediction with calibration and meta-labeling"""
        if not self.models:
            return 0, 0.5, "no_model"
        
        # Prepare features
        X = []
        for feature_name in self.feature_names:
            X.append(features.get(feature_name, 0.0))
        
        X = np.array([X])
        
        # Scale features
        if 'main' in self.scalers:
            X_scaled = self.scalers['main'].transform(X)
        else:
            X_scaled = X
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)[0]
                predictions[name] = pred
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)[0]
                    if len(proba) > 1:
                        probabilities[name] = proba[1] if model.classes_[1] == 1 else proba[0]
                    else:
                        probabilities[name] = proba[0]
                else:
                    probabilities[name] = 0.6 if pred == 1 else 0.4
                    
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
                predictions[name] = 0
                probabilities[name] = 0.5
        
        # Ensemble prediction (majority vote)
        if not predictions:
            # Fallback when no models work
            ensemble_pred = 0
            ensemble_proba = 0.5
        else:
            ensemble_pred = max(set(predictions.values()), key=list(predictions.values()).count)
            ensemble_proba = np.mean(list(probabilities.values()))
        
        # Apply calibration
        if use_calibration:
            calibrated_proba = self._calibrate_probability(ensemble_proba, ensemble_pred)
        else:
            calibrated_proba = ensemble_proba
        
        # Apply meta-labeling
        final_signal, meta_confidence = self.meta_labeling(ensemble_pred, features)
        
        # Determine confidence level
        if calibrated_proba > 0.7:
            confidence = "high"
        elif calibrated_proba > 0.55:
            confidence = "medium"
        else:
            confidence = "low"
        
        return final_signal, calibrated_proba, confidence
    
    def predict_with_confidence(self, quotes: List[Dict]) -> Dict[str, Any]:
        """Predict with confidence for trading signals - main interface for the bot"""
        try:
            # Extract features
            features = self.extract_advanced_features(quotes)
            
            # Make prediction
            signal, probability, confidence = self.predict(features)
            
            return {
                'signal': signal,
                'probability': probability,
                'confidence': confidence,
                'features': features
            }
            
        except Exception as e:
            print(f"Error in predict_with_confidence: {e}")
            # Fallback to basic technical analysis
            return {
                'signal': 0,
                'probability': 0.5,
                'confidence': 'low',
                'features': {}
            }
    
    def _calibrate_probability(self, probability: float, prediction: int) -> float:
        """Calibrate probability using isotonic regression"""
        try:
            if 'random_forest_isotonic' in self.calibrators:
                calibrated = self.calibrators['random_forest_isotonic'].predict([probability])[0]
                return max(0.0, min(1.0, calibrated))
        except Exception as e:
            print(f"Error in probability calibration: {e}")
        
        return probability
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[int]:
        """Make batch predictions"""
        predictions = []
        
        for features in features_list:
            pred, _, _ = self.predict(features, use_calibration=False)
            predictions.append(pred)
        
        return predictions
    
    def calculate_expected_value(self, signal: int, probability: float, 
                              entry_price: float, stop_loss: float, take_profit: float) -> float:
        """Calculate expected value of a trade"""
        if signal == 0:
            return 0.0
        
        # Calculate risk and reward
        if signal == 1:  # Long
            risk = abs(entry_price - stop_loss) / entry_price
            reward = abs(take_profit - entry_price) / entry_price
        else:  # Short
            risk = abs(entry_price - take_profit) / entry_price
            reward = abs(entry_price - stop_loss) / entry_price
        
        # Expected value calculation
        win_probability = probability if signal == 1 else (1 - probability)
        loss_probability = 1 - win_probability
        
        ev = win_probability * reward - loss_probability * risk
        
        return ev
    
    def save_models(self):
        """Save trained models"""
        try:
            # Create models directory
            os.makedirs('models', exist_ok=True)
            
            # Save models
            for name, model in self.models.items():
                with open(f'models/{name}.pkl', 'wb') as f:
                    pickle.dump(model, f)
            
            # Save scalers
            for name, scaler in self.scalers.items():
                with open(f'models/scaler_{name}.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
            
            # Save calibrators
            for name, calibrator in self.calibrators.items():
                with open(f'models/calibrator_{name}.pkl', 'wb') as f:
                    pickle.dump(calibrator, f)
            
            # Save meta-model
            if self.meta_model:
                with open('models/meta_model.pkl', 'wb') as f:
                    pickle.dump(self.meta_model, f)
            
            # Save feature names
            with open('models/feature_names.json', 'w') as f:
                json.dump(self.feature_names, f)
                
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self):
        """Load trained models"""
        try:
            # Load models
            model_files = ['random_forest', 'gradient_boost', 'neural_network']
            for model_file in model_files:
                model_path = f'models/{model_file}.pkl'
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[model_file] = pickle.load(f)
            
            # Load scalers
            scaler_files = ['main']
            for scaler_file in scaler_files:
                scaler_path = f'models/scaler_{scaler_file}.pkl'
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scalers[scaler_file] = pickle.load(f)
            
            # Load calibrators
            calibrator_files = ['random_forest_isotonic', 'random_forest_sigmoid']
            for calibrator_file in calibrator_files:
                calibrator_path = f'models/calibrator_{calibrator_file}.pkl'
                if os.path.exists(calibrator_path):
                    with open(calibrator_path, 'rb') as f:
                        self.calibrators[calibrator_file] = pickle.load(f)
            
            # Load meta-model
            meta_model_path = 'models/meta_model.pkl'
            if os.path.exists(meta_model_path):
                with open(meta_model_path, 'rb') as f:
                    self.meta_model = pickle.load(f)
            
            # Load feature names
            feature_names_path = 'models/feature_names.json'
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'r') as f:
                    self.feature_names = json.load(f)
                    
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def record_signal(self, pair: str, timeframe: str, signal_type: str, probability: float,
                     confidence: str, entry_price: float, stop_loss: float, take_profit: float,
                     expected_value: float, market_regime: str, features: Dict[str, float]):
        """Record a trading signal in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ai_signals 
                (pair, timeframe, signal_type, probability, confidence, entry_price, 
                 stop_loss, take_profit, expected_value, market_regime, features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pair, timeframe, signal_type, probability, confidence, entry_price,
                stop_loss, take_profit, expected_value, market_regime, json.dumps(features)
            ))
            
            signal_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return signal_id
            
        except Exception as e:
            print(f"Error recording signal: {e}")
            return None
    
    def update_signal_outcome(self, signal_id: int, outcome: str, pnl: float):
        """Update signal outcome in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE ai_signals 
                SET outcome = ?, pnl = ? 
                WHERE id = ?
            ''', (outcome, pnl, signal_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error updating signal outcome: {e}")
    
    def get_performance_metrics(self, days: int = 30, pair: str = None, market_regime: str = None) -> Dict[str, float]:
        """Get performance metrics from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query
            query = '''
                SELECT signal_type, probability, confidence, outcome, pnl, market_regime
                FROM ai_signals 
                WHERE timestamp >= datetime('now', '-{} days')
            '''.format(days)
            
            if pair:
                query += f" AND pair = '{pair}'"
            
            if market_regime:
                query += f" AND market_regime = '{market_regime}'"
            
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return {
                    'total_signals': 0,
                    'win_rate': 0.0,
                    'avg_pnl': 0.0,
                    'avg_probability': 0.0,
                    'profit_factor': 0.0,
                    'expected_value': 0.0
                }
            
            # Calculate metrics
            total_signals = len(results)
            winning_trades = sum(1 for r in results if r[3] == 'win')
            losing_trades = sum(1 for r in results if r[3] == 'loss')
            
            win_rate = winning_trades / total_signals if total_signals > 0 else 0.0
            
            pnls = [r[4] for r in results if r[4] is not None]
            avg_pnl = np.mean(pnls) if pnls else 0.0
            
            probabilities = [r[1] for r in results if r[1] is not None]
            avg_probability = np.mean(probabilities) if probabilities else 0.0
            
            # Profit factor
            gross_profit = sum(pnl for pnl in pnls if pnl > 0)
            gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Expected value
            expected_value = avg_pnl * win_rate if win_rate > 0 else 0.0
            
            return {
                'total_signals': total_signals,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'avg_probability': avg_probability,
                'profit_factor': profit_factor,
                'expected_value': expected_value,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades
            }
            
        except Exception as e:
            print(f"Error getting performance metrics: {e}")
            return {
                'total_signals': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'avg_probability': 0.0,
                'profit_factor': 0.0,
                'expected_value': 0.0
            }
    
    def get_market_regime_distribution(self, days: int = 30) -> Dict[str, int]:
        """Get distribution of market regimes"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT market_regime, COUNT(*) as count
                FROM ai_signals 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY market_regime
            '''.format(days))
            
            results = cursor.fetchall()
            conn.close()
            
            return {row[0]: row[1] for row in results}
            
        except Exception as e:
            print(f"Error getting regime distribution: {e}")
            return {}
    
    def record_model_metric(self, model_name: str, metric_name: str, metric_value: float,
                           period_days: int = None, market_regime: str = None):
        """Record model performance metric"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_metrics 
                (model_name, metric_name, metric_value, period_days, market_regime)
                VALUES (?, ?, ?, ?, ?)
            ''', (model_name, metric_name, metric_value, period_days, market_regime))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error recording model metric: {e}")