import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Optional
import io
import base64
from PIL import Image
import pandas as pd
from forex_indicators import SimpleIndicators

class ChartGenerator:
    """Генератор реальных торговых графиков для Telegram бота"""
    
    def __init__(self):
        plt.style.use('dark_background')
        self.colors = {
            'bullish': '#00ff00',
            'bearish': '#ff4444',
            'neutral': '#ffffff',
            'background': '#1a1a1a',
            'grid': '#333333'
        }
    
    def create_candlestick_chart(self, quotes: List[Dict], pair: str, timeframe: str = '1h') -> bytes:
        """Создает свечной график"""
        if len(quotes) < 20:
            return None
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]}, 
                                       facecolor=self.colors['background'])
        
        # Подготовка данных
        times = [datetime.fromtimestamp(q['timestamp']) for q in quotes[-60:]]
        opens = [q['open'] for q in quotes[-60:]]
        highs = [q['high'] for q in quotes[-60:]]
        lows = [q['low'] for q in quotes[-60:]]
        closes = [q['close'] for q in quotes[-60:]]
        volumes = [q['volume'] for q in quotes[-60:]]
        
        # Создание свечей
        for i in range(len(times)):
            color = self.colors['bullish'] if closes[i] > opens[i] else self.colors['bearish']
            ax1.plot([times[i], times[i]], [lows[i], highs[i]], color=color, linewidth=1)
            ax1.plot([times[i], times[i]], [opens[i], closes[i]], color=color, linewidth=3)
        
        # Индикаторы
        self._add_indicators(ax1, times, closes)
        
        # Настройка основного графика
        ax1.set_title(f'{pair} - {timeframe} Chart', color='white', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price', color='white')
        ax1.grid(True, alpha=0.3, color=self.colors['grid'])
        ax1.tick_params(colors='white')
        
        # Объем
        ax2.bar(times, volumes, color=self.colors['neutral'], alpha=0.7)
        ax2.set_ylabel('Volume', color='white')
        ax2.grid(True, alpha=0.3, color=self.colors['grid'])
        ax2.tick_params(colors='white')
        
        # Форматирование времени
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Конвертация в bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                   facecolor=self.colors['background'])
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()
    
    def create_technical_chart(self, quotes: List[Dict], pair: str, 
                               signal_data: Optional[Dict] = None) -> bytes:
        """Создает технический график с индикаторами"""
        if len(quotes) < 50:
            return None
            
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), 
                                gridspec_kw={'height_ratios': [3, 1, 1]},
                                facecolor=self.colors['background'])
        
        ax_price, ax_rsi, ax_macd = axes
        
        # Подготовка данных
        times = [datetime.fromtimestamp(q['timestamp']) for q in quotes[-100:]]
        closes = [q['close'] for q in quotes[-100:]]
        highs = [q['high'] for q in quotes[-100:]]
        lows = [q['low'] for q in quotes[-100:]]
        
        # Индикаторы
        rsi_values = SimpleIndicators.rsi(closes, 14)
        macd_data = SimpleIndicators.macd(closes, 12, 26, 9)
        sma_20 = SimpleIndicators.sma(closes, 20)
        sma_50 = SimpleIndicators.sma(closes, 50)
        
        # Ценовой график
        ax_price.plot(times, closes, color=self.colors['neutral'], linewidth=2, label='Close')
        
        if sma_20:
            ax_price.plot(times[-len(sma_20):], sma_20, color='#ffaa00', 
                         linewidth=1.5, alpha=0.8, label='SMA 20')
        if sma_50:
            ax_price.plot(times[-len(sma_50):], sma_50, color='#ff6600', 
                         linewidth=1.5, alpha=0.8, label='SMA 50')
        
        # Уровни сигнала
        if signal_data:
            self._add_signal_levels(ax_price, times, signal_data)
        
        ax_price.set_title(f'{pair} - Technical Analysis', color='white', 
                          fontsize=14, fontweight='bold')
        ax_price.set_ylabel('Price', color='white')
        ax_price.grid(True, alpha=0.3, color=self.colors['grid'])
        ax_price.legend(loc='upper left')
        ax_price.tick_params(colors='white')
        
        # RSI
        if rsi_values:
            ax_rsi.plot(times[-len(rsi_values):], rsi_values, color='#00aaff', linewidth=2)
            ax_rsi.axhline(y=70, color=self.colors['bearish'], linestyle='--', alpha=0.7)
            ax_rsi.axhline(y=30, color=self.colors['bullish'], linestyle='--', alpha=0.7)
            ax_rsi.fill_between(times[-len(rsi_values):], 70, 100, 
                               color=self.colors['bearish'], alpha=0.2)
            ax_rsi.fill_between(times[-len(rsi_values):], 0, 30, 
                               color=self.colors['bullish'], alpha=0.2)
            ax_rsi.set_ylabel('RSI', color='white')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.grid(True, alpha=0.3, color=self.colors['grid'])
            ax_rsi.tick_params(colors='white')
        
        # MACD
        if macd_data['macd'] and macd_data['signal']:
            macd_times = times[-len(macd_data['macd']):]
            ax_macd.plot(macd_times, macd_data['macd'], color='#00ffaa', 
                        linewidth=2, label='MACD')
            ax_macd.plot(macd_times, macd_data['signal'], color='#ffaa00', 
                        linewidth=2, label='Signal')
            
            # Гистограмма
            histogram = [macd_data['macd'][i] - macd_data['signal'][i] 
                        for i in range(len(macd_data['macd']))]
            colors_hist = [self.colors['bullish'] if h > 0 else self.colors['bearish'] 
                          for h in histogram]
            ax_macd.bar(macd_times, histogram, color=colors_hist, alpha=0.6)
            
            ax_macd.set_ylabel('MACD', color='white')
            ax_macd.set_xlabel('Time', color='white')
            ax_macd.grid(True, alpha=0.3, color=self.colors['grid'])
            ax_macd.legend(loc='upper left')
            ax_macd.tick_params(colors='white')
        
        # Форматирование
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Конвертация в bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor=self.colors['background'])
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()
    
    def _add_indicators(self, ax, times, closes):
        """Добавляет технические индикаторы на график"""
        # Bollinger Bands
        bb_data = SimpleIndicators.bollinger_bands(closes, 20)
        if bb_data['upper'] and bb_data['lower']:
            ax.plot(times[-len(bb_data['upper']):], bb_data['upper'], 
                   color='#ffaa00', linewidth=1, alpha=0.7, linestyle='--')
            ax.plot(times[-len(bb_data['middle']):], bb_data['middle'], 
                   color='#ffffff', linewidth=1, alpha=0.7)
            ax.plot(times[-len(bb_data['lower']):], bb_data['lower'], 
                   color='#00ffaa', linewidth=1, alpha=0.7, linestyle='--')
            
            # Заливка между полосами
            ax.fill_between(times[-len(bb_data['upper']):], 
                           bb_data['upper'], bb_data['lower'], 
                           alpha=0.1, color='#ffffff')
    
    def _add_signal_levels(self, ax, times, signal_data):
        """Добавляет уровни сигнала на график"""
        if signal_data.get('entry_price'):
            ax.axhline(y=signal_data['entry_price'], color='#00ff00', 
                      linewidth=2, linestyle='-', alpha=0.8, 
                      label=f'Entry: {signal_data["entry_price"]:.5f}')
        
        if signal_data.get('stop_loss'):
            ax.axhline(y=signal_data['stop_loss'], color='#ff4444', 
                      linewidth=2, linestyle='--', alpha=0.8, 
                      label=f'SL: {signal_data["stop_loss"]:.5f}')
        
        if signal_data.get('take_profit'):
            ax.axhline(y=signal_data['take_profit'], color='#44ff44', 
                      linewidth=2, linestyle='--', alpha=0.8, 
                      label=f'TP: {signal_data["take_profit"]:.5f}')
    
    def create_signal_preview(self, quotes: List[Dict], pair: str, 
                            signal_data: Dict, evaluation: Dict) -> bytes:
        """Создает превью сигнала с ключевой информацией"""
        chart_bytes = self.create_technical_chart(quotes, pair, signal_data)
        
        if chart_bytes:
            # Добавляем текстовую информацию на изображение
            img = Image.open(io.BytesIO(chart_bytes))
            
            # Создаем новое изображение с местом для текста
            new_height = img.height + 150
            new_img = Image.new('RGB', (img.width, new_height), self.colors['background'])
            new_img.paste(img, (0, 0))
            
            # Здесь можно добавить текст с помощью PIL, но для простоты вернем оригинал
            buf = io.BytesIO()
            new_img.save(buf, format='PNG')
            buf.seek(0)
            return buf.getvalue()
        
        return chart_bytes