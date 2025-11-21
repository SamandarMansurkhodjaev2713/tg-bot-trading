#!/usr/bin/env python3
"""
Простой и легкий Telegram-бот для торговли форексом
Без pandas и сложных зависимостей
"""

import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
from decimal import Decimal
import statistics

import yfinance as yf
import requests
import feedparser
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Валютные пары и их спецификации для Exness
CURRENCY_PAIRS = {
    'XAUUSD': {'spread': 0.2, 'commission': 0.0, 'swap_long': -2.5, 'swap_short': 0.5, 'leverage': 100},
    'EURUSD': {'spread': 0.1, 'commission': 0.0, 'swap_long': -0.8, 'swap_short': 0.2, 'leverage': 500},
    'GBPUSD': {'spread': 0.2, 'commission': 0.0, 'swap_long': -1.0, 'swap_short': 0.3, 'leverage': 500},
    'USDJPY': {'spread': 0.2, 'commission': 0.0, 'swap_long': 0.1, 'swap_short': -0.9, 'leverage': 500},
    'USDCHF': {'spread': 0.3, 'commission': 0.0, 'swap_long': 0.2, 'swap_short': -1.1, 'leverage': 500},
    'AUDUSD': {'spread': 0.2, 'commission': 0.0, 'swap_long': -0.6, 'swap_short': 0.1, 'leverage': 500},
    'USDCAD': {'spread': 0.2, 'commission': 0.0, 'swap_long': -0.4, 'swap_short': -0.3, 'leverage': 500},
    'NZDUSD': {'spread': 0.3, 'commission': 0.0, 'swap_long': -0.5, 'swap_short': 0.1, 'leverage': 500}
}

# Таймфреймы и их периоды в днях
TIMEFRAMES = {
    '1m': 1/24/60,    # 1 минута
    '5m': 1/24/12,    # 5 минут
    '15m': 1/24/4,    # 15 минут
    '1h': 1/24,       # 1 час
    '4h': 1/6,        # 4 часа
    '1d': 1           # 1 день
}

class ForexDatabase:
    """Простая SQLite база данных для хранения данных"""
    
    def __init__(self, db_path: str = "forex_bot.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Инициализация базы данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Таблица для хранения котировок
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                UNIQUE(pair, timeframe, timestamp)
            )
        ''')
        
        # Таблица для хранения сигналов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                signal TEXT NOT NULL,
                probability REAL NOT NULL,
                indicators TEXT NOT NULL,
                news_sentiment REAL
            )
        ''')
        
        # Таблица для хранения результатов бэктестинга
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                start_date DATETIME NOT NULL,
                end_date DATETIME NOT NULL,
                total_trades INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                profit_factor REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                total_return REAL NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_quotes(self, pair: str, timeframe: str, quotes: List[Dict]):
        """Сохранение котировок в базу данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for quote in quotes:
            cursor.execute('''
                INSERT OR REPLACE INTO quotes (pair, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (pair, timeframe, quote['timestamp'], quote['open'], quote['high'], 
                  quote['low'], quote['close'], quote['volume']))
        
        conn.commit()
        conn.close()
    
    def get_quotes(self, pair: str, timeframe: str, limit: int = 100) -> List[Dict]:
        """Получение котировок из базы данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, open, high, low, close, volume
            FROM quotes 
            WHERE pair = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (pair, timeframe, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        quotes = []
        for row in reversed(rows):
            quotes.append({
                'timestamp': row[0],
                'open': row[1],
                'high': row[2],
                'low': row[3],
                'close': row[4],
                'volume': row[5]
            })
        
        return quotes

class SimpleIndicators:
    """Простые технические индикаторы без pandas"""
    
    @staticmethod
    def sma(prices: List[float], period: int) -> List[float]:
        """Простое скользящее среднее"""
        if len(prices) < period:
            return []
        
        sma_values = []
        for i in range(period - 1, len(prices)):
            sma_values.append(sum(prices[i - period + 1:i + 1]) / period)
        
        return sma_values
    
    @staticmethod
    def ema(prices: List[float], period: int) -> List[float]:
        """Экспоненциальное скользящее среднее"""
        if len(prices) < period:
            return []
        
        multiplier = 2 / (period + 1)
        ema_values = [sum(prices[:period]) / period]  # Первое значение - это SMA
        
        for i in range(period, len(prices)):
            ema_values.append((prices[i] - ema_values[-1]) * multiplier + ema_values[-1])
        
        return ema_values
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[float]:
        """Индекс относительной силы"""
        if len(prices) < period + 1:
            return []
        
        rsi_values = []
        gains = []
        losses = []
        
        # Рассчитываем первые изменения
        for i in range(1, period + 1):
            change = prices[i] - prices[i - 1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        for i in range(period + 1, len(prices)):
            change = prices[i] - prices[i - 1]
            gain = max(change, 0)
            loss = max(-change, 0)
            
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))
        
        return rsi_values
    
    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[float]]:
        """MACD индикатор"""
        ema_fast = SimpleIndicators.ema(prices, fast)
        ema_slow = SimpleIndicators.ema(prices, slow)
        
        if len(ema_fast) != len(ema_slow):
            min_len = min(len(ema_fast), len(ema_slow))
            ema_fast = ema_fast[-min_len:]
            ema_slow = ema_slow[-min_len:]
        
        macd_line = [fast - slow for fast, slow in zip(ema_fast, ema_slow)]
        signal_line = SimpleIndicators.ema(macd_line, signal)
        
        # Выравниваем длины
        histogram = []
        if len(signal_line) > 0:
            macd_for_hist = macd_line[-len(signal_line):]
            histogram = [m - s for m, s in zip(macd_for_hist, signal_line)]
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Dict[str, List[float]]:
        """Полосы Боллинджера"""
        sma_values = SimpleIndicators.sma(prices, period)
        
        upper_band = []
        lower_band = []
        
        for i in range(period - 1, len(prices)):
            period_prices = prices[i - period + 1:i + 1]
            std_value = statistics.stdev(period_prices)
            sma_value = sma_values[i - period + 1]
            
            upper_band.append(sma_value + std_dev * std_value)
            lower_band.append(sma_value - std_dev * std_value)
        
        return {
            'upper': upper_band,
            'middle': sma_values,
            'lower': lower_band
        }
    
    @staticmethod
    def atr(quotes: List[Dict], period: int = 14) -> List[float]:
        """Средний истинный диапазон"""
        if len(quotes) < period + 1:
            return []
        
        true_ranges = []
        
        for i in range(1, len(quotes)):
            high = quotes[i]['high']
            low = quotes[i]['low']
            prev_close = quotes[i - 1]['close']
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_ranges.append(max(tr1, tr2, tr3))
        
        # Первое значение ATR - это простое среднее
        atr_values = [sum(true_ranges[:period]) / period]
        
        # Последующие значения - сглаженное среднее
        for i in range(period, len(true_ranges)):
            atr_values.append((atr_values[-1] * (period - 1) + true_ranges[i]) / period)
        
        return atr_values

class NewsService:
    """Сервис для получения новостей"""
    
    def __init__(self):
        self.rss_feeds = [
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://www.reuters.com/markets/rss',
            'https://feeds.forexfactory.com/news.rss',
            'https://www.dailyfx.com/feeds/market-news'
        ]
        self.cache = {}
        self.cache_ttl = 300
    
    def get_forex_news(self, pair: str, limit: int = 5) -> List[Dict]:
        """Получение новостей по валютной паре с кэшированием"""
        now = datetime.utcnow()
        key = (pair, limit)
        cached = self.cache.get(key)
        if cached and (now - cached['time']).total_seconds() < self.cache_ttl:
            return cached['items']
        news_items = []
        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:limit]:
                    title = entry.get('title', '').upper()
                    summary = entry.get('summary', '').upper()
                    pair_currencies = pair[:3] + '/' + pair[3:]
                    if any(curr in title or curr in summary for curr in [pair[:3], pair[3:], pair_currencies]):
                        news_items.append({
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', ''),
                            'published': entry.get('published', ''),
                            'link': entry.get('link', ''),
                            'sentiment': self._analyze_sentiment(entry.get('title', '') + ' ' + entry.get('summary', ''))
                        })
            except Exception as e:
                logger.error(f"Ошибка при получении новостей из {feed_url}: {e}")
                continue
        items = sorted(news_items, key=lambda x: x.get('published', ''), reverse=True)[:limit]
        self.cache[key] = {'time': now, 'items': items}
        return items
    
    def _analyze_sentiment(self, text: str) -> float:
        """Простой анализ тональности текста"""
        positive_words = ['рост', 'подъем', 'роста', 'поднялся', 'вырос', 'укрепился', 'позитив', 'росту', 'подняться']
        negative_words = ['падение', 'снижение', 'упал', 'снизился', 'ослаб', 'негатив', 'снижается', 'падает', 'снижения']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)

class SimpleMLModel:
    """Простая ML модель для прогнозирования"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    def prepare_features(self, quotes: List[Dict]) -> List[List[float]]:
        """Подготовка признаков для модели"""
        features = []
        
        # Извлекаем цены закрытия
        closes = [q['close'] for q in quotes]
        highs = [q['high'] for q in quotes]
        lows = [q['low'] for q in quotes]
        volumes = [q['volume'] for q in quotes]
        
        if len(closes) < 50:  # Недостаточно данных
            return []
        
        # Рассчитываем индикаторы
        sma_10 = SimpleIndicators.sma(closes, 10)
        sma_20 = SimpleIndicators.sma(closes, 20)
        sma_50 = SimpleIndicators.sma(closes, 50)
        
        rsi_14 = SimpleIndicators.rsi(closes, 14)
        
        bb_20 = SimpleIndicators.bollinger_bands(closes, 20)
        
        atr_14 = SimpleIndicators.atr(quotes, 14)
        
        # Создаем признаки для каждой свечи
        for i in range(50, len(closes)):
            feature_vector = []
            
            # Ценовые признаки
            feature_vector.extend([
                closes[i],
                (closes[i] - closes[i-1]) / closes[i-1],  # Изменение цены
                highs[i] / closes[i],  # Отношение high к close
                lows[i] / closes[i],   # Отношение low к close
                volumes[i] / max(volumes[max(0, i-20):i+1]) if max(volumes[max(0, i-20):i+1]) > 0 else 0  # Относительный объем
            ])
            
            # Признаки скользящих средних
            if i >= 49 and len(sma_50) > 0:
                feature_vector.extend([
                    closes[i] / sma_50[-1] if sma_50[-1] > 0 else 0,  # Цена относительно SMA50
                ])
            
            # Признаки RSI
            if i >= 50 + 14 - 1 and len(rsi_14) > 0:
                feature_vector.append(rsi_14[-1] / 100)  # Нормализованный RSI
            
            # Признаки полос Боллинджера
            if i >= 50 + 20 - 1 and len(bb_20['upper']) > 0:
                bb_position = (closes[i] - bb_20['lower'][-1]) / (bb_20['upper'][-1] - bb_20['lower'][-1]) if (bb_20['upper'][-1] - bb_20['lower'][-1]) > 0 else 0.5
                feature_vector.append(bb_position)
            
            # Признаки ATR
            if i >= 50 + 14 - 1 and len(atr_14) > 0:
                feature_vector.append(atr_14[-1] / closes[i])  # ATR как процент от цены
            
            features.append(feature_vector)
        
        return features
    
    def prepare_last_features(self, quotes: List[Dict]) -> List[float]:
        closes = [q['close'] for q in quotes]
        highs = [q['high'] for q in quotes]
        lows = [q['low'] for q in quotes]
        volumes = [q['volume'] for q in quotes]
        if len(closes) < 50:
            return []
        i = len(closes) - 1
        fv = [
            closes[i],
            (closes[i] - closes[i-1]) / closes[i-1] if i > 0 and closes[i-1] != 0 else 0.0,
            highs[i] / closes[i] if closes[i] != 0 else 0.0,
            lows[i] / closes[i] if closes[i] != 0 else 0.0,
            volumes[i] / max(volumes[max(0, i-20):i+1]) if max(volumes[max(0, i-20):i+1]) > 0 else 0.0
        ]
        sma50 = SimpleIndicators.sma(closes[-50:], 50)
        if sma50:
            fv.append(closes[i] / sma50[-1] if sma50[-1] > 0 else 0.0)
        rsi14 = SimpleIndicators.rsi(closes[-15:], 14)
        if rsi14:
            fv.append(rsi14[-1] / 100.0)
        bb20 = SimpleIndicators.bollinger_bands(closes[-20:], 20)
        if bb20['upper'] and bb20['lower'] and (bb20['upper'][-1] - bb20['lower'][-1]) > 0:
            bb_pos = (closes[i] - bb20['lower'][-1]) / (bb20['upper'][-1] - bb20['lower'][-1])
            fv.append(bb_pos)
        atr14 = SimpleIndicators.atr(quotes[-15:], 14)
        if atr14:
            fv.append(atr14[-1] / closes[i] if closes[i] != 0 else 0.0)
        return fv
    
    def prepare_labels(self, quotes: List[Dict], forecast_period: int = 5) -> List[int]:
        """Подготовка меток для обучения (0=держать, 1=купить, -1=продать)"""
        labels = []
        closes = [q['close'] for q in quotes]
        
        for i in range(50, len(closes) - forecast_period):
            future_return = (closes[i + forecast_period] - closes[i]) / closes[i]
            
            if future_return > 0.005:  # Более 0.5% роста
                labels.append(1)  # Покупать
            elif future_return < -0.005:  # Более 0.5% падения
                labels.append(-1)  # Продавать
            else:
                labels.append(0)  # Держать
        
        return labels
    
    def train(self, quotes: List[Dict]) -> bool:
        """Обучение модели"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            features = self.prepare_features(quotes)
            labels = self.prepare_labels(quotes)
            
            if len(features) != len(labels) or len(features) < 10:
                logger.error("Недостаточно данных для обучения")
                return False
            
            # Разделяем данные на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Обучаем модель
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            self.model.fit(X_train, y_train)
            
            # Оцениваем качество
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            logger.info(f"Модель обучена. Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
            self.is_trained = True
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {e}")
            return False
    
    def predict(self, quotes: List[Dict]) -> Dict[str, float]:
        """Предсказание сигнала"""
        if not self.is_trained or not self.model:
            return {'signal': 0, 'probability': 0.5}
        
        try:
            last_features = self.prepare_last_features(quotes)
            if not last_features:
                return {'signal': 0, 'probability': 0.5}
            prediction = self.model.predict([last_features])[0]
            probabilities = self.model.predict_proba([last_features])[0]
            max_probability = max(probabilities)
            return {
                'signal': int(prediction),
                'probability': float(max_probability),
                'probabilities': {
                    'sell': probabilities[0] if len(probabilities) > 0 else 0.33,
                    'hold': probabilities[1] if len(probabilities) > 1 else 0.34,
                    'buy': probabilities[2] if len(probabilities) > 2 else 0.33
                }
            }
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            return {'signal': 0, 'probability': 0.5}

class TradingAIAssistant:
    """AI ассистент для оценки торговых сигналов"""
    
    def __init__(self):
        self.risk_levels = {
            'conservative': {'max_risk': 0.01, 'min_rr_ratio': 2.0, 'max_drawdown': 0.05},
            'moderate': {'max_risk': 0.02, 'min_rr_ratio': 1.5, 'max_drawdown': 0.10},
            'aggressive': {'max_risk': 0.05, 'min_rr_ratio': 1.2, 'max_drawdown': 0.20}
        }
    
    def parse_signal_request(self, text: str) -> Dict:
        """Парсинг торгового сигнала из текста"""
        import re
        
        signal_data = {
            'pair': None,
            'direction': None,
            'entry': None,
            'stop_loss': None,
            'take_profit': None,
            'risk_level': 'moderate',
            'confidence': None,
            'success': False,
            'error': None
        }
        
        # Поиск валютной пары
        pair_pattern = r'(XAUUSD|EURUSD|GBPUSD|USDJPY|USDCHF|AUDUSD|USDCAD|NZDUSD)'
        pair_match = re.search(pair_pattern, text.upper())
        if pair_match:
            signal_data['pair'] = pair_match.group(1)
        
        # Поиск направления сделки
        if any(word in text.lower() for word in ['лонг', 'long', 'buy', 'покупка']):
            signal_data['direction'] = 'long'
        elif any(word in text.lower() for word in ['шорт', 'short', 'sell', 'продажа']):
            signal_data['direction'] = 'short'
        
        # Поиск цен
        price_pattern = r'(\d+(?:\.\d+)?)'
        prices = re.findall(price_pattern, text)
        
        if len(prices) >= 3:
            signal_data['entry'] = float(prices[0])
            signal_data['stop_loss'] = float(prices[1])
            signal_data['take_profit'] = float(prices[2])
        elif len(prices) >= 2:
            signal_data['stop_loss'] = float(prices[0])
            signal_data['take_profit'] = float(prices[1])
        
        # Поиск уровня риска
        if any(word in text.lower() for word in ['консервативный', 'conservative', 'низкий']):
            signal_data['risk_level'] = 'conservative'
        elif any(word in text.lower() for word in ['агрессивный', 'aggressive', 'высокий']):
            signal_data['risk_level'] = 'aggressive'
        
        # Проверяем минимальные требования для успешного парсинга
        if signal_data['pair'] and signal_data['direction'] and signal_data['stop_loss'] and signal_data['take_profit']:
            signal_data['success'] = True
        else:
            signal_data['error'] = 'Недостаточно данных для анализа сигнала'
        
        return signal_data
    
    def analyze_market_conditions(self, quotes: List[Dict]) -> Dict:
        """Анализ текущих рыночных условий"""
        if len(quotes) < 50:
            return {'error': 'Недостаточно данных для анализа'}
        closes = [q['close'] for q in quotes]
        current_price = closes[-1]
        rsi_values = SimpleIndicators.rsi(closes[-15:], 14)
        macd_data = SimpleIndicators.macd(closes[-60:], 12, 26, 9)
        bb_data = SimpleIndicators.bollinger_bands(closes[-20:], 20)
        atr_values = SimpleIndicators.atr(quotes[-15:], 14)
        sma_20 = SimpleIndicators.sma(closes[-20:], 20)
        sma_50 = SimpleIndicators.sma(closes[-50:], 50)
        market_analysis = {
            'current_price': current_price,
            'rsi': rsi_values[-1] if rsi_values else 50,
            'macd_signal': 'bullish' if (macd_data['macd'] and macd_data['signal'] and 
                                        macd_data['macd'][-1] > macd_data['signal'][-1]) else 'bearish',
            'bb_position': (current_price - bb_data['lower'][-1]) / (bb_data['upper'][-1] - bb_data['lower'][-1]) 
                          if bb_data['lower'] and bb_data['upper'] else 0.5,
            'atr': atr_values[-1] if atr_values else current_price * 0.01,
            'trend': 'bullish' if (sma_20 and sma_50 and sma_20[-1] > sma_50[-1]) else 'bearish',
            'volatility': 'high' if (atr_values and atr_values[-1] / current_price > 0.02) else 'low'
        }
        return market_analysis
    
    def evaluate_signal(self, signal_data: Dict, market_analysis: Dict, pair_specs: Dict) -> Dict:
        """Оценка торгового сигнала"""
        if not signal_data['pair'] or not signal_data['direction']:
            return {'error': 'Не удалось распознать валютную пару или направление сделки'}
        
        if not signal_data['stop_loss'] or not signal_data['take_profit']:
            return {'error': 'Не указаны стоп-лосс или тейк-профит'}
        
        current_price = market_analysis['current_price']
        direction = signal_data['direction']
        sl = signal_data['stop_loss']
        tp = signal_data['take_profit']
        
        # Расчет риска и соотношения риск/прибыль
        if direction == 'long':
            risk = current_price - sl
            reward = tp - current_price
            risk_percentage = (risk / current_price) * 100
        else:  # short
            risk = sl - current_price
            reward = current_price - tp
            risk_percentage = (risk / current_price) * 100
        
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Получаем настройки риска
        risk_settings = self.risk_levels[signal_data['risk_level']]
        
        # Анализ сигнала
        score = 0
        feedback = []
        warnings = []
        recommendations = []
        
        # 1. Соотношение риск/прибыль
        if rr_ratio >= risk_settings['min_rr_ratio']:
            score += 2
            feedback.append(f"✅ Хорошее соотношение риск/прибыль: {rr_ratio:.2f}:1")
        elif rr_ratio >= 1.0:
            score += 1
            feedback.append(f"⚠️ Соотношение риск/прибыль: {rr_ratio:.2f}:1 (можно лучше)")
        else:
            feedback.append(f"❌ Плохое соотношение риск/прибыль: {rr_ratio:.2f}:1")
            warnings.append("Тейк-профит слишком близко к входу")
        
        # 2. Размер риска
        if risk_percentage <= risk_settings['max_risk'] * 100:
            score += 2
            feedback.append(f"✅ Приемлемый риск: {risk_percentage:.2f}%")
        else:
            feedback.append(f"❌ Риск слишком высокий: {risk_percentage:.2f}%")
            warnings.append(f"Рекомендуется уменьшить размер позиции")
        
        # 3. Соответствие рыночным условиям
        if direction == 'long' and market_analysis['trend'] == 'bullish':
            score += 2
            feedback.append("✅ Сигнал в направлении тренда")
        elif direction == 'short' and market_analysis['trend'] == 'bearish':
            score += 2
            feedback.append("✅ Сигнал в направлении тренда")
        else:
            feedback.append("⚠️ Сигнал против тренда")
            warnings.append("Рассмотрите возможность отложенного входа")
        
        # 4. RSI анализ
        rsi = market_analysis['rsi']
        if direction == 'long' and rsi < 30:
            score += 2
            feedback.append("✅ RSI в зоне перепроданности - хороший момент для покупки")
        elif direction == 'short' and rsi > 70:
            score += 2
            feedback.append("✅ RSI в зоне перекупленности - хороший момент для продажи")
        elif (direction == 'long' and rsi > 70) or (direction == 'short' and rsi < 30):
            feedback.append("⚠️ RSI указывает на возможный разворот")
            warnings.append("Возможно, слишком поздно для входа")
        
        # 5. MACD подтверждение
        if market_analysis['macd_signal'] == 'bullish' and direction == 'long':
            score += 1
            feedback.append("✅ MACD подтверждает восходящий импульс")
        elif market_analysis['macd_signal'] == 'bearish' and direction == 'short':
            score += 1
            feedback.append("✅ MACD подтверждает нисходящий импульс")
        
        # 6. Bollinger Bands позиция
        bb_pos = market_analysis['bb_position']
        if direction == 'long' and bb_pos < 0.3:
            score += 1
            feedback.append("✅ Цена близка к нижней полосе Боллинджера")
        elif direction == 'short' and bb_pos > 0.7:
            score += 1
            feedback.append("✅ Цена близка к верхней полосе Боллинджера")
        
        # 7. ATR анализ для стоп-лосса
        atr = market_analysis['atr']
        min_sl_distance = atr * 1.5  # Минимальное расстояние для SL
        
        if direction == 'long':
            actual_sl_distance = current_price - sl
        else:
            actual_sl_distance = sl - current_price
        
        if actual_sl_distance >= min_sl_distance:
            score += 1
            feedback.append("✅ Стоп-лосс учитывает волатильность")
        else:
            warnings.append("Стоп-лосс слишком близко, возможен ложный пробой")
        
        # Рекомендации
        if score >= 6:
            recommendation = "🟢 СИЛЬНЫЙ СИГНАЛ - Рекомендуется к исполнению"
        elif score >= 4:
            recommendation = "🟡 УМЕРЕННЫЙ СИГНАЛ - Можно рассмотреть с осторожностью"
        else:
            recommendation = "🔴 СЛАБЫЙ СИГНАЛ - Лучше воздержаться или дождаться подтверждения"
        
        # Дополнительные рекомендации
        if market_analysis['volatility'] == 'high':
            recommendations.append("Высокая волатильность - уменьшите размер позиции")
        
        if rr_ratio < risk_settings['min_rr_ratio']:
            recommendations.append(f"Увеличьте тейк-профит до минимум {risk_settings['min_rr_ratio']}:1")
        
        return {
            'score': score,
            'recommendation': recommendation,
            'feedback': feedback,
            'warnings': warnings,
            'recommendations': recommendations,
            'risk_reward_ratio': rr_ratio,
            'risk_percentage': risk_percentage,
            'market_conditions': market_analysis
        }

class ForexBot:
    """Основной класс Telegram-бота"""
    
    def __init__(self, token: str):
        self.token = token
        self.db = ForexDatabase()
        self.news_service = NewsService()
        self.ml_model = SimpleMLModel()
        self.ai_assistant = TradingAIAssistant()
        self.quotes_cache = {}
        self.application = None
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start"""
        welcome_message = """
🤖 *Добро пожаловать в Forex AI Advisor!*

Я - ваш интеллектуальный помощник для торговли на форекс.

📊 *Доступные команды:*
• /analyze [пара] [таймфрейм] - Анализ валютной пары
• /news [пара] - Последние новости по паре
• /train [пара] [таймфрейм] - Обучение модели
• /backtest [пара] [таймфрейм] [дней] - Бэктестинг
• /chatai [сигнал] - AI оценка вашего торгового сигнала
• /status - Статус бота и моделей

💡 *Примеры использования:*
• `/analyze XAUUSD 1h` - Анализ золота на 1 час
• `/news EURUSD` - Новости по евро/доллару
• `/train XAUUSD 1d` - Обучение модели на дневных данных
• `/backtest EURUSD 4h 30` - Бэктест за 30 дней
• `/chatai Хочу лонг XAUUSD со стопом 2650 и тейком 2720` - AI оценка сигнала

⚠️ *Важно:* Используйте только для образовательных целей!
        """
        
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /analyze"""
        try:
            args = context.args
            if len(args) < 2:
                await update.message.reply_text(
                    "❌ *Ошибка:* Укажите валютную пару и таймфрейм\n"
                    "📌 *Пример:* `/analyze XAUUSD 1h`",
                    parse_mode='Markdown'
                )
                return
            
            pair = args[0].upper()
            timeframe = args[1].lower()
            
            if pair not in CURRENCY_PAIRS:
                available_pairs = ', '.join(CURRENCY_PAIRS.keys())
                await update.message.reply_text(
                    f"❌ *Ошибка:* Неверная валютная пара\n"
                    f"📌 *Доступные пары:* {available_pairs}",
                    parse_mode='Markdown'
                )
                return
            
            if timeframe not in TIMEFRAMES:
                available_tfs = ', '.join(TIMEFRAMES.keys())
                await update.message.reply_text(
                    f"❌ *Ошибка:* Неверный таймфрейм\n"
                    f"📌 *Доступные таймфреймы:* {available_tfs}",
                    parse_mode='Markdown'
                )
                return
            
            # Получаем данные
            await update.message.reply_text(f"📊 *Анализ {pair} {timeframe}...*", parse_mode='Markdown')
            
            quotes = self.get_quotes(pair, timeframe, 100)
            
            if not quotes:
                await update.message.reply_text("❌ *Ошибка:* Не удалось получить данные", parse_mode='Markdown')
                return
            
            # Анализируем данные
            analysis = self.analyze_data(quotes, pair, timeframe)
            
            # Формируем ответ
            response = f"""
📈 *Анализ {pair} {timeframe}*

💰 *Текущая цена:* {analysis['current_price']:.5f}
📊 *Изменение за 24ч:* {analysis['change_24h']:+.2f}%

🔍 *Технические индикаторы:*
• RSI (14): {analysis['rsi']:.1f} {'🟢' if analysis['rsi'] < 30 else '🔴' if analysis['rsi'] > 70 else '⚪'}
• MACD: {'🟢' if analysis['macd_signal'] > 0 else '🔴'}
• BB Position: {analysis['bb_position']:.1%}

🤖 *ML Сигнал:*
• Рекомендация: {analysis['ml_signal']}
• Уверенность: {analysis['ml_probability']:.1%}

⚠️ *Риск-менеджмент:*
• ATR: {analysis['atr']:.5f}
• Рекомендуемый SL: {analysis['stop_loss']:.5f}
• Рекомендуемый TP: {analysis['take_profit']:.5f}

📰 *Последние новости:*
{analysis['news_summary']}
            """
            
            await update.message.reply_text(response.strip(), parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Ошибка в команде analyze: {e}")
            await update.message.reply_text(f"❌ *Ошибка:* {str(e)}", parse_mode='Markdown')
    
    async def news_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /news"""
        try:
            args = context.args
            if len(args) < 1:
                await update.message.reply_text(
                    "❌ *Ошибка:* Укажите валютную пару\n"
                    "📌 *Пример:* `/news EURUSD`",
                    parse_mode='Markdown'
                )
                return
            
            pair = args[0].upper()
            
            if pair not in CURRENCY_PAIRS:
                available_pairs = ', '.join(CURRENCY_PAIRS.keys())
                await update.message.reply_text(
                    f"❌ *Ошибка:* Неверная валютная пара\n"
                    f"📌 *Доступные пары:* {available_pairs}",
                    parse_mode='Markdown'
                )
                return
            
            await update.message.reply_text(f"📰 *Поиск новостей для {pair}...*", parse_mode='Markdown')
            
            news_items = self.news_service.get_forex_news(pair, 5)
            
            if not news_items:
                await update.message.reply_text("📰 *Новости не найдены*", parse_mode='Markdown')
                return
            
            response = f"📰 *Последние новости {pair}:*\n\n"
            
            for i, item in enumerate(news_items, 1):
                sentiment_emoji = "🟢" if item['sentiment'] > 0.1 else "🔴" if item['sentiment'] < -0.1 else "⚪"
                response += f"{i}. *{item['title']}* {sentiment_emoji}\n"
                response += f"   {item['summary'][:100]}...\n"
                response += f"   [Читать далее]({item['link']})\n\n"
            
            await update.message.reply_text(response.strip(), parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Ошибка в команде news: {e}")
            await update.message.reply_text(f"❌ *Ошибка:* {str(e)}", parse_mode='Markdown')
    
    async def train_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /train"""
        try:
            args = context.args
            if len(args) < 2:
                await update.message.reply_text(
                    "❌ *Ошибка:* Укажите валютную пару и таймфрейм\n"
                    "📌 *Пример:* `/train XAUUSD 1d`",
                    parse_mode='Markdown'
                )
                return
            
            pair = args[0].upper()
            timeframe = args[1].lower()
            
            if pair not in CURRENCY_PAIRS:
                available_pairs = ', '.join(CURRENCY_PAIRS.keys())
                await update.message.reply_text(
                    f"❌ *Ошибка:* Неверная валютная пара\n"
                    f"📌 *Доступные пары:* {available_pairs}",
                    parse_mode='Markdown'
                )
                return
            
            if timeframe not in TIMEFRAMES:
                available_tfs = ', '.join(TIMEFRAMES.keys())
                await update.message.reply_text(
                    f"❌ *Ошибка:* Неверный таймфрейм\n"
                    f"📌 *Доступные таймфреймы:* {available_tfs}",
                    parse_mode='Markdown'
                )
                return
            
            await update.message.reply_text(f"🧠 *Обучение модели {pair} {timeframe}...*", parse_mode='Markdown')
            
            # Получаем больше данных для обучения
            quotes = self.get_quotes(pair, timeframe, 500)
            
            if len(quotes) < 100:
                await update.message.reply_text("❌ *Ошибка:* Недостаточно данных для обучения", parse_mode='Markdown')
                return
            
            # Обучаем модель
            success = self.ml_model.train(quotes)
            
            if success:
                await update.message.reply_text(
                    f"✅ *Модель успешно обучена!*\n"
                    f"📊 *Данных использовано:* {len(quotes)} свечей\n"
                    f"⏰ *Период:* {timeframe}",
                    parse_mode='Markdown'
                )
            else:
                await update.message.reply_text("❌ *Ошибка при обучении модели*", parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Ошибка в команде train: {e}")
            await update.message.reply_text(f"❌ *Ошибка:* {str(e)}", parse_mode='Markdown')
    
    async def backtest_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /backtest"""
        try:
            args = context.args
            if len(args) < 3:
                await update.message.reply_text(
                    "❌ *Ошибка:* Укажите валютную пару, таймфрейм и количество дней\n"
                    "📌 *Пример:* `/backtest EURUSD 4h 30`",
                    parse_mode='Markdown'
                )
                return
            
            pair = args[0].upper()
            timeframe = args[1].lower()
            days = int(args[2])
            
            if pair not in CURRENCY_PAIRS:
                available_pairs = ', '.join(CURRENCY_PAIRS.keys())
                await update.message.reply_text(
                    f"❌ *Ошибка:* Неверная валютная пара\n"
                    f"📌 *Доступные пары:* {available_pairs}",
                    parse_mode='Markdown'
                )
                return
            
            if timeframe not in TIMEFRAMES:
                available_tfs = ', '.join(TIMEFRAMES.keys())
                await update.message.reply_text(
                    f"❌ *Ошибка:* Неверный таймфрейм\n"
                    f"📌 *Доступные таймфреймы:* {available_tfs}",
                    parse_mode='Markdown'
                )
                return
            
            await update.message.reply_text(f"📈 *Бэктестинг {pair} {timeframe} за {days} дней...*", parse_mode='Markdown')
            
            # Получаем данные для бэктеста
            limit_quotes = min(days * 24 * 60, 2000)
            quotes = self.get_quotes(pair, timeframe, limit_quotes)
            
            if len(quotes) < 50:
                await update.message.reply_text("❌ *Ошибка:* Недостаточно данных для бэктеста", parse_mode='Markdown')
                return
            
            # Запускаем бэктест
            results = self.run_backtest(quotes, pair, timeframe)
            
            response = f"""
📈 *Результаты бэктеста {pair} {timeframe}*

📊 *Основные метрики:*
• Всего сделок: {results['total_trades']}
• Прибыльных сделок: {results['winning_trades']} ({results['win_rate']:.1%})
• Убыточных сделок: {results['losing_trades']}
• Профит-фактор: {results['profit_factor']:.2f}
• Макс. просадка: {results['max_drawdown']:.2%}
• Общая прибыль: {results['total_return']:.2%}

💰 *Статистика сделок:*
• Средняя прибыль: {results['avg_win']:.5f}
• Средний убыток: {results['avg_loss']:.5f}
• Соотношение прибыль/убыток: {results['win_loss_ratio']:.2f}

📅 *Период:* {results['start_date']} - {results['end_date']}
            """
            
            await update.message.reply_text(response.strip(), parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Ошибка в команде backtest: {e}")
            await update.message.reply_text(f"❌ *Ошибка:* {str(e)}", parse_mode='Markdown')
    
    async def chatai_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /chatai - AI помощник для оценки торговых сигналов"""
        try:
            # Получаем текст запроса
            user_text = ' '.join(context.args) if context.args else ""
            
            if not user_text:
                await update.message.reply_text(
                    "🤖 *AI Помощник по торговым сигналам*\n\n"
                    "📋 *Примеры использования:*\n"
                    "• `/chatai Хочу открыть лонг XAUUSD со стопом 2650 и тейком 2720`\n"
                    "• `/chatai EURUSD шорт от 1.0850, SL 1.0900, TP 1.0750 консервативно`\n"
                    "• `/chaut GBPUSD лонг 1.2450 стоп 1.2400 тейк 1.2550 агрессивно`\n\n"
                    "🔧 *Параметры:*\n"
                    "• Валютная пара: XAUUSD, EURUSD, GBPUSD и др.\n"
                    "• Направление: лонг/лонг, шорт/short, buy/sell\n"
                    "• Уровни: цена входа, стоп-лосс, тейк-профит\n"
                    "• Риск: консервативный, умеренный*, агрессивный\n\n"
                    "⚠️ *Важно:* Используйте точные данные для лучшей оценки",
                    parse_mode='Markdown'
                )
                return
            
            await update.message.reply_text("🤖 *AI анализирует ваш сигнал...*", parse_mode='Markdown')
            
            # Парсим сигнал
            signal_data = self.ai_assistant.parse_signal_request(user_text)
            
            if not signal_data['pair']:
                await update.message.reply_text(
                    "❌ *Ошибка:* Не удалось распознать валютную пару.\n"
                    "📌 *Пример:* `XAUUSD`, `EURUSD`, `GBPUSD`",
                    parse_mode='Markdown'
                )
                return
            
            # Получаем рыночные данные для анализа
            quotes = self.get_quotes(signal_data['pair'], '1h', 100)  # Используем 1h для анализа
            
            if not quotes:
                await update.message.reply_text(
                    "❌ *Ошибка:* Не удалось получить рыночные данные для анализа",
                    parse_mode='Markdown'
                )
                return
            
            # Анализируем рыночные условия
            market_analysis = self.ai_assistant.analyze_market_conditions(quotes)
            
            if 'error' in market_analysis:
                await update.message.reply_text(
                    f"❌ *Ошибка:* {market_analysis['error']}",
                    parse_mode='Markdown'
                )
                return
            
            # Получаем спецификации пары
            pair_specs = CURRENCY_PAIRS.get(signal_data['pair'], {})
            
            # Оцениваем сигнал
            evaluation = self.ai_assistant.evaluate_signal(signal_data, market_analysis, pair_specs)
            
            if 'error' in evaluation:
                await update.message.reply_text(
                    f"❌ *Ошибка:* {evaluation['error']}",
                    parse_mode='Markdown'
                )
                return
            
            # Формируем ответ
            current_price = market_analysis['current_price']
            
            response = f"""
🤖 *AI Оценка торгового сигнала*

📊 *Ваш сигнал:*
• Пара: {signal_data['pair']}
• Направление: {'📈 ЛОНГ' if signal_data['direction'] == 'long' else '📉 ШОРТ'}
• Текущая цена: {current_price:.5f}
• Стоп-лосс: {signal_data['stop_loss']:.5f}
• Тейк-профит: {signal_data['take_profit']:.5f}
• Уровень риска: {signal_data['risk_level']}

🎯 *Оценка AI:*
{evaluation['recommendation']}

📈 *Рыночные условия:*
• Тренд: {'🟢 Восходящий' if market_analysis['trend'] == 'bullish' else '🔴 Нисходящий'}
• RSI: {market_analysis['rsi']:.1f} {'🔴' if market_analysis['rsi'] > 70 else '🟢' if market_analysis['rsi'] < 30 else '⚪'}
• MACD: {'🟢 Бычий' if market_analysis['macd_signal'] == 'bullish' else '🔴 Медвежий'}
• Волатильность: {'🔴 Высокая' if market_analysis['volatility'] == 'high' else '🟢 Низкая'}

💡 *Анализ:*
"""
            
            # Добавляем обратную связь
            for feedback in evaluation['feedback']:
                response += f"• {feedback}\n"
            
            # Предупреждения
            if evaluation['warnings']:
                response += f"\n⚠️ *Предупреждения:*\n"
                for warning in evaluation['warnings']:
                    response += f"• {warning}\n"
            
            # Рекомендации
            if evaluation['recommendations']:
                response += f"\n🔧 *Рекомендации:*\n"
                for rec in evaluation['recommendations']:
                    response += f"• {rec}\n"
            
            # Технические детали
            response += f"""
📊 *Технические детали:*
• Соотношение риск/прибыль: {evaluation['risk_reward_ratio']:.2f}:1
• Риск от депозита: {evaluation['risk_percentage']:.2f}%
• Оценка качества: {evaluation['score']}/10
"""
            
            # Добавляем новости
            news_items = self.news_service.get_forex_news(signal_data['pair'], 2)
            if news_items:
                response += f"\n📰 *Последние новости {signal_data['pair']}:*\n"
                for item in news_items:
                    sentiment_emoji = "🟢" if item['sentiment'] > 0.1 else "🔴" if item['sentiment'] < -0.1 else "⚪"
                    response += f"• {item['title'][:60]}... {sentiment_emoji}\n"
            
            response += f"""

⚠️ *Важно:*
• Это образовательный анализ, не финансовый совет
• Всегда используйте дополнительное подтверждение
• Тестируйте стратегии на демо-счете
• Управляйте рисками разумно
"""
            
            await update.message.reply_text(response.strip(), parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Ошибка в команде chatai: {e}")
            await update.message.reply_text(
                f"❌ *Ошибка при анализе сигнала:* {str(e)}\n"
                f"📌 Проверьте формат ввода и попробуйте снова",
                parse_mode='Markdown'
            )
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /status"""
        try:
            status_text = f"""
🤖 *Статус Forex AI Advisor*

📅 *Время запуска:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🧠 *Модель:*
• Обучена: {'✅' if self.ml_model.is_trained else '❌'}
• Тип: RandomForestClassifier
• Признаков: 10+

📊 *Доступные пары:* {len(CURRENCY_PAIRS)}
• {', '.join(list(CURRENCY_PAIRS.keys())[:4])}
• {', '.join(list(CURRENCY_PAIRS.keys())[4:])}

⏰ *Таймфреймы:* {len(TIMEFRAMES)}
• {', '.join(TIMEFRAMES.keys())}

📰 *Новостные источники:* {len(self.news_service.rss_feeds)}
• Bloomberg, Reuters, ForexFactory, DailyFX

⚙️ *Настройки риска:*
• Макс. риск на сделку: 2%
• ATR множитель: 2.0
• Мин. соотношение прибыли/убытка: 1.5:1
            """
            
            await update.message.reply_text(status_text.strip(), parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Ошибка в команде status: {e}")
            await update.message.reply_text(f"❌ *Ошибка:* {str(e)}", parse_mode='Markdown')
    
    def get_quotes(self, pair: str, timeframe: str, limit: int = 100) -> List[Dict]:
        """Получение котировок с кэшированием"""
        try:
            key = (pair, timeframe)
            now = datetime.utcnow()
            ttl_map = {'1m': 30, '5m': 120, '15m': 300, '1h': 600, '4h': 1800, '1d': 43200}
            ttl = ttl_map.get(timeframe, 300)
            cached = self.quotes_cache.get(key)
            if cached and (now - cached['time']).total_seconds() < ttl and cached['quotes']:
                return cached['quotes'][-limit:]
            yahoo_pair = self.pair_to_yahoo_format(pair)
            period_map = {
                '1m': f"{max(1, limit//60)}d",
                '5m': f"{max(1, limit//12)}d",
                '15m': f"{max(1, limit//4)}d",
                '1h': f"{max(1, limit//24)}d",
                '4h': f"{max(1, limit//6)}d",
                '1d': f"{max(30, limit)}d"
            }
            period = period_map.get(timeframe, f"{limit}d")
            ticker = yf.Ticker(yahoo_pair)
            interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '1h', '1d': '1d'}
            interval = interval_map.get(timeframe, '1d')
            hist = ticker.history(period=period, interval=interval)
            if hist.empty:
                logger.error(f"Нет данных для {yahoo_pair}")
                return []
            quotes = []
            for index, row in hist.iterrows():
                quotes.append({
                    'timestamp': index.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            self.db.save_quotes(pair, timeframe, quotes)
            self.quotes_cache[key] = {'time': now, 'quotes': quotes}
            return quotes[-limit:]
        except Exception as e:
            logger.error(f"Ошибка при получении котировок для {pair}: {e}")
            return self.db.get_quotes(pair, timeframe, limit)
    
    def pair_to_yahoo_format(self, pair: str) -> str:
        """Преобразование пары в формат Yahoo Finance"""
        # XAUUSD -> GC=F (золото фьючерсы)
        if pair == 'XAUUSD':
            return 'GC=F'
        elif pair == 'XAGUSD':  # Серебро
            return 'SI=F'
        else:
            # Обычные валютные пары
            return pair[:3] + pair[3:] + "=X"
    
    def analyze_data(self, quotes: List[Dict], pair: str, timeframe: str) -> Dict:
        """Анализ данных и расчет индикаторов"""
        if len(quotes) < 50:
            return {}
        
        closes = [q['close'] for q in quotes]
        highs = [q['high'] for q in quotes]
        lows = [q['low'] for q in quotes]
        
        # Текущая цена и изменение
        current_price = closes[-1]
        price_24h_ago = closes[-min(24*60, len(closes))] if timeframe == '1m' else closes[-min(24, len(closes))]
        change_24h = (current_price - price_24h_ago) / price_24h_ago * 100
        
        # Технические индикаторы
        rsi_values = SimpleIndicators.rsi(closes, 14)
        rsi = rsi_values[-1] if rsi_values else 50
        
        macd_data = SimpleIndicators.macd(closes, 12, 26, 9)
        macd_signal = 1 if macd_data['macd'] and macd_data['signal'] and macd_data['macd'][-1] > macd_data['signal'][-1] else -1
        
        bb_data = SimpleIndicators.bollinger_bands(closes, 20)
        bb_position = (current_price - bb_data['lower'][-1]) / (bb_data['upper'][-1] - bb_data['lower'][-1]) if bb_data['lower'] and bb_data['upper'] else 0.5
        
        # ML сигнал
        ml_prediction = self.ml_model.predict(quotes)
        ml_signal_map = {-1: "🔴 ПРОДАВАТЬ", 0: "⚪ ДЕРЖАТЬ", 1: "🟢 ПОКУПАТЬ"}
        ml_signal = ml_signal_map.get(ml_prediction['signal'], "⚪ ДЕРЖАТЬ")
        ml_probability = ml_prediction['probability']
        
        # Риск-менеджмент
        atr_values = SimpleIndicators.atr(quotes, 14)
        atr = atr_values[-1] if atr_values else current_price * 0.01
        
        stop_loss = current_price - 2 * atr if ml_prediction['signal'] == 1 else current_price + 2 * atr
        take_profit = current_price + 3 * atr if ml_prediction['signal'] == 1 else current_price - 3 * atr
        
        # Новости
        news_items = self.news_service.get_forex_news(pair, 3)
        news_summary = ""
        if news_items:
            for item in news_items[:2]:
                sentiment_emoji = "🟢" if item['sentiment'] > 0.1 else "🔴" if item['sentiment'] < -0.1 else "⚪"
                news_summary += f"• {item['title'][:50]}... {sentiment_emoji}\n"
        else:
            news_summary = "• Новости не найдены"
        
        return {
            'current_price': current_price,
            'change_24h': change_24h,
            'rsi': rsi,
            'macd_signal': macd_signal,
            'bb_position': bb_position,
            'ml_signal': ml_signal,
            'ml_probability': ml_probability,
            'atr': atr,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'news_summary': news_summary.strip()
        }
    
    def run_backtest(self, quotes: List[Dict], pair: str, timeframe: str) -> Dict:
        """Простой бэктестинг"""
        if len(quotes) < 100:
            return {}
        
        trades = []
        position = 0  # 0 = нет позиции, 1 = лонг, -1 = шорт
        entry_price = 0
        
        # Проходим по данным с шагом
        for i in range(50, len(quotes) - 5):
            current_quotes = quotes[:i+1]
            
            # Получаем сигнал от ML модели
            prediction = self.ml_model.predict(current_quotes)
            signal = prediction['signal']
            
            current_price = quotes[i]['close']
            
            # Логика входа в сделку
            if position == 0 and signal != 0:
                position = signal
                entry_price = current_price
                
                trades.append({
                    'entry_date': quotes[i]['timestamp'],
                    'entry_price': entry_price,
                    'position': position,
                    'signal_probability': prediction['probability']
                })
            
            # Логика выхода из сделки
            elif position != 0:
                # Выходим через 5 свечей или при смене сигнала
                if i >= len(quotes) - 5 or (signal != 0 and signal != position):
                    exit_price = current_price
                    pnl = (exit_price - entry_price) * position / entry_price
                    
                    trades[-1].update({
                        'exit_date': quotes[i]['timestamp'],
                        'exit_price': exit_price,
                        'pnl': pnl
                    })
                    
                    position = 0
        
        # Рассчитываем статистику
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'total_return': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'win_loss_ratio': 0,
                'start_date': quotes[0]['timestamp'][:10],
                'end_date': quotes[-1]['timestamp'][:10]
            }
        
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        total_return = sum(t.get('pnl', 0) for t in trades)
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        avg_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Рассчитываем максимальную просадку
        cumulative_returns = []
        current_return = 0
        max_return = 0
        max_drawdown = 0
        
        for trade in trades:
            current_return += trade.get('pnl', 0)
            cumulative_returns.append(current_return)
            max_return = max(max_return, current_return)
            current_drawdown = max_return - current_return
            max_drawdown = max(max_drawdown, current_drawdown)
        
        # Профит-фактор
        gross_profit = sum(t.get('pnl', 0) for t in winning_trades)
        gross_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'start_date': quotes[0]['timestamp'][:10],
            'end_date': quotes[-1]['timestamp'][:10]
        }
    
    def run(self):
        """Запуск бота"""
        logger.info("Запуск Forex AI Advisor...")
        
        # Создаем приложение
        self.application = Application.builder().token(self.token).build()
        
        # Регистрируем обработчики команд
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("analyze", self.analyze_command))
        self.application.add_handler(CommandHandler("news", self.news_command))
        self.application.add_handler(CommandHandler("train", self.train_command))
        self.application.add_handler(CommandHandler("backtest", self.backtest_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("chatai", self.chatai_command))
        
        # Запускаем бота
        logger.info("Бот запущен и готов к работе!")
        self.application.run_polling()

def main():
    """Главная функция"""
    # Получаем токен из переменных окружения
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN не найден в переменных окружения")
        print("❌ Ошибка: Установите переменную окружения TELEGRAM_BOT_TOKEN")
        print("Пример: set TELEGRAM_BOT_TOKEN=ваш_токен")
        return
    
    # Создаем и запускаем бота
    bot = ForexBot(token)
    bot.run()

if __name__ == '__main__':
    main()