import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Optional, Tuple
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from forex_indicators import AdvancedTechnicalIndicators
import seaborn as sns

class AdvancedChartGenerator:
    """Advanced chart generator with 3-panel professional charts for Telegram bot"""
    
    def __init__(self):
        plt.style.use('dark_background')
        self.colors = {
            'bullish': '#00ff00',
            'bearish': '#ff4444', 
            'neutral': '#ffffff',
            'background': '#0d1117',  # GitHub dark theme
            'grid': '#30363d',
            'signal_buy': '#238636',
            'signal_sell': '#da3633',
            'text_primary': '#f0f6fc',
            'text_secondary': '#8b949e'
        }
        self.indicators = AdvancedTechnicalIndicators()
        
        # Set matplotlib parameters for professional appearance
        plt.rcParams.update({
            'figure.facecolor': self.colors['background'],
            'axes.facecolor': self.colors['background'],
            'axes.edgecolor': self.colors['grid'],
            'axes.labelcolor': self.colors['text_primary'],
            'xtick.color': self.colors['text_secondary'],
            'ytick.color': self.colors['text_secondary'],
            'grid.color': self.colors['grid'],
            'legend.facecolor': self.colors['background'],
            'legend.edgecolor': self.colors['grid'],
            'text.color': self.colors['text_primary'],
            'font.family': 'monospace',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })
    
    def create_professional_chart(self, quotes: List[Dict], pair: str, timeframe: str, 
                                 signal_data: Optional[Dict] = None, 
                                 ai_analysis: Optional[Dict] = None) -> bytes:
        """Create professional 3-panel chart with price, RSI, and MACD"""
        if len(quotes) < 30:
            return None
            
        # Create figure with 3 panels
        fig = plt.figure(figsize=(16, 12), facecolor=self.colors['background'])
        
        # Define grid layout
        gs = fig.add_gridspec(3, 1, height_ratios=[4, 1.5, 1.5], hspace=0.05)
        ax_price = fig.add_subplot(gs[0])
        ax_rsi = fig.add_subplot(gs[1], sharex=ax_price)
        ax_macd = fig.add_subplot(gs[2], sharex=ax_price)
        
        # Prepare data
        times = [datetime.fromtimestamp(q['timestamp']) for q in quotes]
        opens = np.array([q['open'] for q in quotes])
        highs = np.array([q['high'] for q in quotes])
        lows = np.array([q['low'] for q in quotes])
        closes = np.array([q['close'] for q in quotes])
        volumes = np.array([q['volume'] for q in quotes])
        
        # Calculate indicators
        indicators_data = self._calculate_all_indicators(quotes)
        
        # Panel 1: Price and volume
        self._create_price_panel(ax_price, times, opens, highs, lows, closes, volumes, 
                                indicators_data, pair, timeframe, signal_data)
        
        # Panel 2: RSI
        self._create_rsi_panel(ax_rsi, times, indicators_data, signal_data)
        
        # Panel 3: MACD
        self._create_macd_panel(ax_macd, times, indicators_data, signal_data)
        
        # Add signal annotations if available
        if signal_data:
            self._add_signal_annotations(fig, ax_price, times, signal_data, ai_analysis)
        
        # Format x-axis
        self._format_x_axis(ax_price, times, timeframe)
        
        # Add title with AI analysis summary
        self._add_chart_title(fig, pair, timeframe, ai_analysis, signal_data)
        
        # Convert to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight',
                   facecolor=self.colors['background'], edgecolor='none',
                   pad_inches=0.1)
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()
    
    def _calculate_all_indicators(self, quotes: List[Dict]) -> Dict[str, np.ndarray]:
        """Calculate all technical indicators"""
        closes = np.array([q['close'] for q in quotes])
        highs = np.array([q['high'] for q in quotes])
        lows = np.array([q['low'] for q in quotes])
        
        # RSI
        rsi_values = self.indicators.rsi(closes.tolist(), period=14)
        rsi_array = np.array(rsi_values) if rsi_values else np.full(len(closes), 50)
        
        # MACD
        macd_data = self.indicators.macd(closes.tolist(), fast=12, slow=26, signal=9)
        macd_array = np.array(macd_data.get('macd', [0] * len(closes)))
        signal_array = np.array(macd_data.get('signal', [0] * len(closes)))
        histogram_array = np.array(macd_data.get('histogram', [0] * len(closes)))
        
        # Bollinger Bands
        bb_data = self.indicators.bollinger_bands(closes.tolist(), period=20, std_dev=2)
        bb_upper = np.array(bb_data.get('upper', closes))
        bb_middle = np.array(bb_data.get('middle', closes))
        bb_lower = np.array(bb_data.get('lower', closes))
        
        # Moving averages
        sma_20 = np.array(self.indicators.sma(closes.tolist(), period=20) or closes)
        sma_50 = np.array(self.indicators.sma(closes.tolist(), period=50) or closes)
        ema_12 = np.array(self.indicators.ema(closes.tolist(), period=12) or closes)
        
        # ATR for volatility
        atr_values = self.indicators.atr(highs.tolist(), lows.tolist(), closes.tolist(), period=14)
        atr_array = np.array(atr_values) if atr_values else np.full(len(closes), 0.001)
        
        # Advanced indicators
        adx_data = self.indicators.adx_dmi([{'high': h, 'low': l, 'close': c} 
                                           for h, l, c in zip(highs, lows, closes)], period=14)
        adx_array = np.array(adx_data.get('adx', [20] * len(closes)))
        
        # Market regime
        regime_data = self.indicators.market_regime([{'close': c} for c in closes], period=20)
        
        return {
            'rsi': rsi_array,
            'macd': macd_array,
            'macd_signal': signal_array,
            'macd_histogram': histogram_array,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'ema_12': ema_12,
            'atr': atr_array,
            'adx': adx_array,
            'regime': regime_data
        }
    
    def _create_price_panel(self, ax, times, opens, highs, lows, closes, volumes,
                           indicators_data, pair, timeframe, signal_data):
        """Create the price panel with candlesticks and indicators"""
        # Plot candlesticks
        self._plot_candlesticks(ax, times, opens, highs, lows, closes)
        
        # Add moving averages
        ax.plot(times, indicators_data['sma_20'], color='#ffaa00', linewidth=1.5, 
                alpha=0.8, label='SMA 20')
        ax.plot(times, indicators_data['sma_50'], color='#ff6600', linewidth=1.5, 
                alpha=0.8, label='SMA 50')
        ax.plot(times, indicators_data['ema_12'], color='#00aaff', linewidth=1.5, 
                alpha=0.8, label='EMA 12')
        
        # Add Bollinger Bands
        ax.plot(times, indicators_data['bb_upper'], color='#ffffff', linewidth=1, 
                alpha=0.6, linestyle='--', label='BB Upper')
        ax.plot(times, indicators_data['bb_lower'], color='#ffffff', linewidth=1, 
                alpha=0.6, linestyle='--', label='BB Lower')
        
        # Fill Bollinger Band area
        ax.fill_between(times, indicators_data['bb_upper'], indicators_data['bb_lower'], 
                         alpha=0.1, color='#ffffff')
        
        # Add signal levels
        if signal_data:
            self._add_signal_levels_to_price(ax, signal_data)
        
        # Volume subplot
        ax_vol = ax.twinx()
        colors = [self.colors['bullish'] if c > o else self.colors['bearish'] 
                 for c, o in zip(closes, opens)]
        ax_vol.bar(times, volumes, alpha=0.3, color=colors, width=0.8)
        ax_vol.set_ylabel('Volume', color=self.colors['text_secondary'], fontsize=9)
        ax_vol.tick_params(colors=self.colors['text_secondary'], labelsize=8)
        
        # Styling
        ax.set_ylabel('Price', color=self.colors['text_primary'], fontsize=10)
        ax.grid(True, alpha=0.2, color=self.colors['grid'])
        ax.legend(loc='upper left', framealpha=0.8, facecolor=self.colors['background'])
        ax.tick_params(colors=self.colors['text_secondary'], labelsize=8)
        
        # Set y-axis to show reasonable price range
        price_range = max(closes) - min(closes)
        ax.set_ylim(min(closes) - price_range * 0.1, max(closes) + price_range * 0.1)
    
    def _create_rsi_panel(self, ax, times, indicators_data, signal_data):
        """Create the RSI panel"""
        rsi_values = indicators_data['rsi']
        
        # Plot RSI
        ax.plot(times, rsi_values, color='#00aaff', linewidth=2, label='RSI')
        
        # Add overbought/oversold levels
        ax.axhline(y=70, color=self.colors['bearish'], linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=30, color=self.colors['bullish'], linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=50, color=self.colors['text_secondary'], linestyle='-', alpha=0.3, linewidth=0.5)
        
        # Fill overbought/oversold areas
        ax.fill_between(times, 70, 100, alpha=0.15, color=self.colors['bearish'])
        ax.fill_between(times, 0, 30, alpha=0.15, color=self.colors['bullish'])
        
        # Add signal markers if available
        if signal_data and 'rsi_signal' in signal_data:
            signal_times = [times[i] for i in range(len(times)) if i < len(signal_data['rsi_signal'])]
            signal_values = signal_data['rsi_signal'][:len(times)]
            ax.scatter(signal_times, signal_values, color='#ffff00', s=20, alpha=0.8, zorder=5)
        
        # Styling
        ax.set_ylabel('RSI', color=self.colors['text_primary'], fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.2, color=self.colors['grid'])
        ax.legend(loc='upper left', framealpha=0.8, facecolor=self.colors['background'])
        ax.tick_params(colors=self.colors['text_secondary'], labelsize=8)
        
        # Remove x-axis labels for RSI panel
        ax.tick_params(axis='x', labelbottom=False)
    
    def _create_macd_panel(self, ax, times, indicators_data, signal_data):
        """Create the MACD panel"""
        macd_values = indicators_data['macd']
        signal_values = indicators_data['macd_signal']
        histogram_values = indicators_data['macd_histogram']
        
        # Plot MACD lines
        ax.plot(times, macd_values, color='#00ffaa', linewidth=2, label='MACD')
        ax.plot(times, signal_values, color='#ffaa00', linewidth=2, label='Signal')
        
        # Plot histogram
        colors = [self.colors['bullish'] if h > 0 else self.colors['bearish'] 
                 for h in histogram_values]
        ax.bar(times, histogram_values, color=colors, alpha=0.6, width=0.8)
        
        # Add zero line
        ax.axhline(y=0, color=self.colors['text_secondary'], linestyle='-', alpha=0.5, linewidth=1)
        
        # Add signal markers if available
        if signal_data and 'macd_signal' in signal_data:
            signal_times = [times[i] for i in range(len(times)) if i < len(signal_data['macd_signal'])]
            signal_values = signal_data['macd_signal'][:len(times)]
            ax.scatter(signal_times, signal_values, color='#ffff00', s=20, alpha=0.8, zorder=5)
        
        # Styling
        ax.set_ylabel('MACD', color=self.colors['text_primary'], fontsize=10)
        ax.set_xlabel('Time', color=self.colors['text_primary'], fontsize=10)
        ax.grid(True, alpha=0.2, color=self.colors['grid'])
        ax.legend(loc='upper left', framealpha=0.8, facecolor=self.colors['background'])
        ax.tick_params(colors=self.colors['text_secondary'], labelsize=8)
    
    def _plot_candlesticks(self, ax, times, opens, highs, lows, closes):
        """Plot candlesticks with professional styling"""
        # Calculate candle colors
        colors = []
        for i in range(len(closes)):
            if closes[i] > opens[i]:
                colors.append(self.colors['bullish'])
            elif closes[i] < opens[i]:
                colors.append(self.colors['bearish'])
            else:
                colors.append(self.colors['neutral'])
        
        # Plot wicks (high-low lines)
        for i in range(len(times)):
            ax.plot([times[i], times[i]], [lows[i], highs[i]], 
                   color=colors[i], linewidth=1, alpha=0.8)
        
        # Plot bodies (open-close rectangles)
        width = (times[1] - times[0]) * 0.6 if len(times) > 1 else 0.01
        for i in range(len(times)):
            height = abs(closes[i] - opens[i])
            bottom = min(opens[i], closes[i])
            
            if height == 0:  # Doji candle
                ax.plot([times[i] - width/2, times[i] + width/2], 
                       [closes[i], closes[i]], color=colors[i], linewidth=2)
            else:
                rect = plt.Rectangle((times[i] - width/2, bottom), width, height,
                                   facecolor=colors[i], edgecolor=colors[i], alpha=0.8)
                ax.add_patch(rect)
    
    def _add_signal_levels_to_price(self, ax, signal_data):
        """Add signal entry/stop/take profit levels to price panel"""
        if signal_data.get('entry_price'):
            ax.axhline(y=signal_data['entry_price'], color=self.colors['signal_buy'], 
                      linewidth=2, linestyle='-', alpha=0.9,
                      label=f"Entry: {signal_data['entry_price']:.5f}")
        
        if signal_data.get('stop_loss'):
            ax.axhline(y=signal_data['stop_loss'], color=self.colors['bearish'], 
                      linewidth=2, linestyle='--', alpha=0.9,
                      label=f"SL: {signal_data['stop_loss']:.5f}")
        
        if signal_data.get('take_profit'):
            ax.axhline(y=signal_data['take_profit'], color=self.colors['bullish'], 
                      linewidth=2, linestyle='--', alpha=0.9,
                      label=f"TP: {signal_data['take_profit']:.5f}")
    
    def _add_signal_annotations(self, fig, ax_price, times, signal_data, ai_analysis):
        """Add professional signal annotations to the chart"""
        if not signal_data or not ai_analysis:
            return
            
        # Find the signal time (most recent time for now)
        signal_time = times[-1]
        signal_price = signal_data.get('entry_price', signal_data.get('current_price', 0))
        
        # Create annotation text
        annotation_text = self._create_annotation_text(signal_data, ai_analysis)
        
        # Add text box
        bbox_props = dict(
            boxstyle="round,pad=0.5",
            facecolor=self.colors['background'],
            edgecolor=self.colors['grid'],
            alpha=0.9,
            linewidth=1
        )
        
        # Position annotation in upper right corner
        ax_price.text(0.98, 0.98, annotation_text, 
                     transform=ax_price.transAxes,
                     fontsize=9, color=self.colors['text_primary'],
                     verticalalignment='top', horizontalalignment='right',
                     bbox=bbox_props, fontfamily='monospace')
        
        # Add arrow pointing to signal level
        if signal_data.get('entry_price'):
            ax_price.annotate('Signal', 
                            xy=(signal_time, signal_data['entry_price']),
                            xytext=(signal_time, signal_data['entry_price'] + 
                                   (max(ax_price.get_ylim()) - min(ax_price.get_ylim())) * 0.1),
                            arrowprops=dict(arrowstyle='->', color=self.colors['signal_buy'], 
                                          alpha=0.8, lw=1.5),
                            fontsize=8, color=self.colors['signal_buy'],
                            ha='center')
    
    def _create_annotation_text(self, signal_data: Dict, ai_analysis: Dict) -> str:
        """Create professional annotation text"""
        direction = signal_data.get('direction', 'Unknown').upper()
        confidence = ai_analysis.get('confidence', 0) * 100
        expected_value = ai_analysis.get('expected_value', 0)
        
        # Color based on direction
        if direction == 'BUY':
            direction_color = '🟢'
        elif direction == 'SELL':
            direction_color = '🔴'
        else:
            direction_color = '⚪'
        
        text = f"""{direction_color} SIGNAL: {direction}
Confidence: {confidence:.1f}%
Expected Value: ${expected_value:.2f}
Entry: {signal_data.get('entry_price', 'N/A'):.5f}
SL: {signal_data.get('stop_loss', 'N/A'):.5f}
TP: {signal_data.get('take_profit', 'N/A'):.5f}
Risk/Reward: {signal_data.get('risk_reward', 'N/A'):.2f}"""
        
        return text
    
    def _format_x_axis(self, ax, times, timeframe):
        """Format x-axis based on timeframe"""
        # Set major and minor locators based on timeframe
        if 'm' in timeframe:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
            ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif 'h' in timeframe:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        elif 'd' in timeframe:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # Rotate labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Set x-axis limits with some padding
        time_range = max(times) - min(times)
        padding = time_range * 0.05
        ax.set_xlim(min(times) - padding, max(times) + padding)
    
    def _add_chart_title(self, fig, pair, timeframe, ai_analysis, signal_data):
        """Add comprehensive chart title"""
        title_text = f"{pair} - {timeframe} Chart"
        
        if ai_analysis:
            confidence = ai_analysis.get('confidence', 0) * 100
            expected_value = ai_analysis.get('expected_value', 0)
            title_text += f" | AI Confidence: {confidence:.1f}% | EV: ${expected_value:.2f}"
        
        if signal_data:
            direction = signal_data.get('direction', 'Unknown')
            title_text += f" | Signal: {direction.upper()}"
        
        fig.suptitle(title_text, fontsize=16, fontweight='bold', 
                    color=self.colors['text_primary'], y=0.98)
    
    def create_signal_summary_chart(self, quotes: List[Dict], pair: str, 
                                  signal_analysis: Dict, ai_prediction: Dict) -> bytes:
        """Create a comprehensive signal summary chart"""
        if len(quotes) < 20:
            return None
            
        # Create main chart
        main_chart = self.create_professional_chart(
            quotes, pair, '1h', 
            signal_data=signal_analysis,
            ai_analysis=ai_prediction
        )
        
        if not main_chart:
            return None
            
        # Open main chart as PIL image
        img = Image.open(io.BytesIO(main_chart))
        
        # Add summary panel at the bottom
        summary_height = 200
        new_img = Image.new('RGB', (img.width, img.height + summary_height), 
                           self.colors['background'])
        new_img.paste(img, (0, 0))
        
        # Add summary text
        draw = ImageDraw.Draw(new_img)
        
        # Try to use a better font if available
        try:
            font = ImageFont.truetype("arial.ttf", 12)
            font_bold = ImageFont.truetype("arialbd.ttf", 14)
        except:
            font = ImageFont.load_default()
            font_bold = font
        
        # Summary text
        summary_text = self._create_summary_text(signal_analysis, ai_prediction)
        
        # Draw text with professional formatting
        y_offset = img.height + 20
        x_offset = 20
        
        for line in summary_text.split('\n'):
            if line.strip():
                if line.startswith('📊') or line.startswith('🎯') or line.startswith('⚡'):
                    draw.text((x_offset, y_offset), line, font=font_bold, 
                             fill=self.colors['text_primary'])
                else:
                    draw.text((x_offset, y_offset), line, font=font, 
                             fill=self.colors['text_secondary'])
                y_offset += 20
        
        # Convert back to bytes
        buf = io.BytesIO()
        new_img.save(buf, format='PNG', quality=95, optimize=True)
        buf.seek(0)
        
        return buf.getvalue()
    
    def _create_summary_text(self, signal_analysis: Dict, ai_prediction: Dict) -> str:
        """Create comprehensive summary text"""
        direction = signal_analysis.get('direction', 'Unknown')
        entry = signal_analysis.get('entry_price', 0)
        sl = signal_analysis.get('stop_loss', 0)
        tp = signal_analysis.get('take_profit', 0)
        
        confidence = ai_prediction.get('confidence', 0) * 100
        expected_value = ai_prediction.get('expected_value', 0)
        win_probability = ai_prediction.get('win_probability', 0) * 100
        
        # Risk/Reward calculation
        if direction == 'BUY':
            risk = entry - sl
            reward = tp - entry
        else:  # SELL
            risk = sl - entry
            reward = entry - tp
            
        risk_reward = reward / risk if risk > 0 else 0
        
        summary = f"""📊 SIGNAL ANALYSIS SUMMARY
Direction: {direction.upper()} | Confidence: {confidence:.1f}% | Win Prob: {win_probability:.1f}%

🎯 TRADE PARAMETERS
Entry: {entry:.5f} | Stop Loss: {sl:.5f} | Take Profit: {tp:.5f}
Risk: ${risk:.2f} | Reward: ${reward:.2f} | R:R Ratio: {risk_reward:.2f}:1

⚡ AI PREDICTION
Expected Value: ${expected_value:.2f} | Signal Strength: {'Strong' if confidence > 70 else 'Moderate' if confidence > 50 else 'Weak'}
Recommended Position Size: {ai_prediction.get('position_size', 'N/A')} units"""
        
        return summary
    
    def create_backtest_chart(self, quotes: List[Dict], trades: List[Dict], 
                            pair: str, performance_metrics: Dict) -> bytes:
        """Create backtesting performance chart"""
        if len(quotes) < 50 or not trades:
            return None
            
        fig, (ax_price, ax_equity) = plt.subplots(2, 1, figsize=(16, 10), 
                                                  height_ratios=[3, 2],
                                                  facecolor=self.colors['background'])
        
        times = [datetime.fromtimestamp(q['timestamp']) for q in quotes]
        closes = np.array([q['close'] for q in quotes])
        
        # Price panel with trade markers
        ax_price.plot(times, closes, color=self.colors['neutral'], linewidth=2, 
                     alpha=0.8, label='Price')
        
        # Add trade markers
        for trade in trades:
            trade_time = datetime.fromtimestamp(trade['timestamp'])
            entry_price = trade['entry_price']
            direction = trade['direction']
            pnl = trade.get('pnl', 0)
            
            color = self.colors['bullish'] if pnl > 0 else self.colors['bearish']
            marker = '^' if direction == 'BUY' else 'v'
            
            ax_price.scatter(trade_time, entry_price, color=color, marker=marker, 
                           s=100, alpha=0.8, zorder=5)
            
            # Add exit marker
            if trade.get('exit_time'):
                exit_time = datetime.fromtimestamp(trade['exit_time'])
                exit_price = trade.get('exit_price', entry_price)
                ax_price.scatter(exit_time, exit_price, color=color, marker='x', 
                               s=80, alpha=0.8, zorder=5)
        
        # Equity curve panel
        equity_times = [datetime.fromtimestamp(t['timestamp']) for t in trades]
        equity_values = []
        current_equity = 10000  # Starting equity
        
        for trade in trades:
            current_equity += trade.get('pnl', 0)
            equity_values.append(current_equity)
        
        ax_equity.plot(equity_times, equity_values, color='#00ff00', linewidth=2, 
                      label='Equity Curve')
        ax_equity.axhline(y=10000, color=self.colors['text_secondary'], 
                           linestyle='--', alpha=0.5, label='Starting Equity')
        
        # Add performance metrics
        if performance_metrics:
            final_equity = equity_values[-1] if equity_values else 10000
            total_return = (final_equity - 10000) / 10000 * 100
            win_rate = performance_metrics.get('win_rate', 0) * 100
            
            metrics_text = f"""Total Return: {total_return:.2f}%
Win Rate: {win_rate:.1f}%
Final Equity: ${final_equity:.2f}"""
            
            ax_equity.text(0.02, 0.98, metrics_text, transform=ax_equity.transAxes,
                          fontsize=10, color=self.colors['text_primary'],
                          verticalalignment='top', bbox=dict(
                              boxstyle="round,pad=0.3",
                              facecolor=self.colors['background'],
                              edgecolor=self.colors['grid'],
                              alpha=0.8
                          ))
        
        # Styling
        ax_price.set_ylabel('Price', color=self.colors['text_primary'])
        ax_price.grid(True, alpha=0.2)
        ax_price.legend()
        
        ax_equity.set_ylabel('Equity ($)', color=self.colors['text_primary'])
        ax_equity.set_xlabel('Time', color=self.colors['text_primary'])
        ax_equity.grid(True, alpha=0.2)
        ax_equity.legend()
        
        # Format x-axis
        self._format_x_axis(ax_equity, times, '1h')
        
        plt.tight_layout()
        
        # Convert to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight',
                   facecolor=self.colors['background'])
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()
    
    def create_multi_timeframe_chart(self, timeframe_data: Dict[str, List[Dict]], 
                                   pair: str, signal_data: Optional[Dict] = None) -> bytes:
        """Create multi-timeframe analysis chart"""
        fig, axes = plt.subplots(len(timeframe_data), 1, figsize=(16, 4 * len(timeframe_data)),
                                 facecolor=self.colors['background'])
        
        if len(timeframe_data) == 1:
            axes = [axes]
        
        for i, (timeframe, quotes) in enumerate(timeframe_data.items()):
            if len(quotes) < 20:
                continue
                
            ax = axes[i]
            times = [datetime.fromtimestamp(q['timestamp']) for q in quotes]
            closes = [q['close'] for q in quotes]
            
            # Plot price
            ax.plot(times, closes, color=self.colors['neutral'], linewidth=2, 
                   alpha=0.8, label='Close')
            
            # Add moving averages
            sma_20 = self.indicators.sma(closes, period=20)
            if sma_20:
                sma_times = times[-len(sma_20):]
                ax.plot(sma_times, sma_20, color='#ffaa00', linewidth=1.5, 
                       alpha=0.7, label='SMA 20')
            
            # Add signal levels if provided
            if signal_data:
                self._add_signal_levels_to_price(ax, signal_data)
            
            # Styling
            ax.set_title(f'{pair} - {timeframe}', color=self.colors['text_primary'])
            ax.set_ylabel('Price', color=self.colors['text_primary'])
            ax.grid(True, alpha=0.2)
            ax.legend(loc='upper left')
            
            # Format x-axis
            self._format_x_axis(ax, times, timeframe)
        
        plt.tight_layout()
        
        # Convert to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight',
                   facecolor=self.colors['background'])
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()