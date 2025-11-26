import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from loguru import logger
import json

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    initial_balance: float = 10000.0
    spread_pips: float = 1.5  # Typical forex spread
    commission_per_lot: float = 7.0  # Commission per lot in USD
    slippage_pips: float = 0.5  # Slippage per trade
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_positions: int = 5
    min_position_size: float = 0.01
    max_position_size: float = 1.0
    leverage: int = 100

@dataclass
class TradeResult:
    """Result of a single backtested trade"""
    entry_time: datetime
    exit_time: datetime
    pair: str
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    position_size: float
    pips: float
    profit_loss: float
    commission: float
    spread_cost: float
    slippage_cost: float
    holding_period: int  # in minutes
    max_favorable_excursion: float  # MFE in pips
    max_adverse_excursion: float  # MAE in pips
    risk_reward_ratio: float
    signal_confidence: float
    market_regime: str

@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    total_return: float
    max_drawdown: float
    max_drawdown_duration: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    expectancy: float
    recovery_factor: float
    consecutive_wins: int
    consecutive_losses: int
    avg_holding_period: float
    avg_risk_reward: float
    final_balance: float
    total_profit: float
    total_commission: float
    total_spread_cost: float
    total_slippage_cost: float
    monthly_returns: Dict[str, float]
    yearly_returns: Dict[str, float]
    signal_accuracy_by_confidence: Dict[str, float]
    performance_by_market_regime: Dict[str, Dict[str, float]]
    performance_by_hour: Dict[int, float]
    performance_by_day_of_week: Dict[str, float]

class AdvancedBacktestingEngine:
    """Advanced backtesting engine with realistic market simulation"""
    
    def __init__(self, config: BacktestConfig, db_path: str = "backtesting.db"):
        self.config = config
        self.db_path = db_path
        self.trades: List[TradeResult] = []
        self.equity_curve: List[float] = []
        self.setup_database()
        
    def setup_database(self):
        """Initialize backtesting database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_time TEXT,
                exit_time TEXT,
                pair TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                position_size REAL,
                pips REAL,
                profit_loss REAL,
                commission REAL,
                spread_cost REAL,
                slippage_cost REAL,
                holding_period INTEGER,
                max_favorable_excursion REAL,
                max_adverse_excursion REAL,
                risk_reward_ratio REAL,
                signal_confidence REAL,
                market_regime TEXT,
                backtest_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_sessions (
                id TEXT PRIMARY KEY,
                start_time TEXT,
                end_time TEXT,
                initial_balance REAL,
                final_balance REAL,
                total_trades INTEGER,
                win_rate REAL,
                profit_factor REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                config TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def calculate_pip_value(self, pair: str, price: float, position_size: float) -> float:
        """Calculate pip value for different currency pairs"""
        if 'JPY' in pair:
            return (0.01 / price) * position_size * 100000
        else:
            return (0.0001 / price) * position_size * 100000
            
    def apply_spread(self, price: float, pair: str, direction: str) -> float:
        """Apply realistic spread to entry/exit prices"""
        pip_size = 0.01 if 'JPY' in pair else 0.0001
        spread_cost = self.config.spread_pips * pip_size
        
        if direction == 'long':
            return price + spread_cost  # Buy at ask
        else:
            return price - spread_cost  # Sell at bid
            
    def apply_slippage(self, price: float, pair: str, volatility: float) -> float:
        """Apply slippage based on market volatility"""
        pip_size = 0.01 if 'JPY' in pair else 0.0001
        
        # Increase slippage in high volatility
        volatility_multiplier = 1.0 + (volatility * 2)
        slippage_pips = self.config.slippage_pips * volatility_multiplier
        slippage_cost = slippage_pips * pip_size
        
        # Random direction for slippage (can be positive or negative)
        slippage_direction = np.random.choice([-1, 1])
        return price + (slippage_cost * slippage_direction)
        
    def calculate_commission(self, position_size: float) -> float:
        """Calculate commission based on position size"""
        lots = position_size
        return lots * self.config.commission_per_lot
        
    def simulate_trade_execution(self, signal: Dict, market_data: List[Dict]) -> Optional[TradeResult]:
        """Simulate realistic trade execution with all market frictions"""
        try:
            entry_time = datetime.fromisoformat(signal['timestamp'])
            pair = signal['pair']
            direction = signal['direction']
            entry_price = float(signal['entry_price'])
            stop_loss = float(signal['stop_loss'])
            take_profit = float(signal['take_profit'])
            confidence = float(signal.get('confidence', 0.7))
            market_regime = signal.get('market_regime', 'unknown')
            
            # Find entry point in market data
            entry_idx = None
            for i, data in enumerate(market_data):
                if datetime.fromisoformat(data['timestamp']) >= entry_time:
                    entry_idx = i
                    break
                    
            if entry_idx is None:
                return None
                
            # Apply spread and slippage to entry
            real_entry_price = self.apply_spread(entry_price, pair, direction)
            
            # Calculate volatility for slippage calculation
            recent_data = market_data[max(0, entry_idx-20):entry_idx+1]
            if len(recent_data) > 1:
                prices = [float(d['close']) for d in recent_data]
                volatility = np.std(np.diff(prices)) / np.mean(prices)
            else:
                volatility = 0.001
                
            real_entry_price = self.apply_slippage(real_entry_price, pair, volatility)
            
            # Calculate position size based on risk
            risk_amount = self.config.initial_balance * self.config.risk_per_trade
            stop_distance_pips = abs(real_entry_price - stop_loss) / (0.01 if 'JPY' in pair else 0.0001)
            pip_value = self.calculate_pip_value(pair, real_entry_price, 1.0)
            position_size = min(
                risk_amount / (stop_distance_pips * pip_value),
                self.config.max_position_size
            )
            
            # Simulate trade progression
            max_favorable_excursion = 0
            max_adverse_excursion = 0
            exit_price = None
            exit_time = None
            exit_reason = None
            
            for i in range(entry_idx + 1, len(market_data)):
                current_data = market_data[i]
                current_time = datetime.fromisoformat(current_data['timestamp'])
                current_price = float(current_data['high'])
                current_low = float(current_data['low'])
                
                # Check stop loss and take profit
                if direction == 'long':
                    # Long position checks
                    if current_low <= stop_loss:
                        exit_price = stop_loss
                        exit_time = current_time
                        exit_reason = 'stop_loss'
                        break
                    elif current_price >= take_profit:
                        exit_price = take_profit
                        exit_time = current_time
                        exit_reason = 'take_profit'
                        break
                        
                    # Track excursions
                    favorable_pips = (current_price - real_entry_price) / (0.01 if 'JPY' in pair else 0.0001)
                    adverse_pips = (real_entry_price - current_low) / (0.01 if 'JPY' in pair else 0.0001)
                    
                else:  # Short position
                    if current_price >= stop_loss:
                        exit_price = stop_loss
                        exit_time = current_time
                        exit_reason = 'stop_loss'
                        break
                    elif current_low <= take_profit:
                        exit_price = take_profit
                        exit_time = current_time
                        exit_reason = 'take_profit'
                        break
                        
                    # Track excursions
                    favorable_pips = (real_entry_price - current_price) / (0.01 if 'JPY' in pair else 0.0001)
                    adverse_pips = (float(current_data['high']) - real_entry_price) / (0.01 if 'JPY' in pair else 0.0001)
                
                max_favorable_excursion = max(max_favorable_excursion, favorable_pips)
                max_adverse_excursion = max(max_adverse_excursion, adverse_pips)
                
                # Time-based exit (after 24 hours)
                if (current_time - entry_time).total_seconds() > 86400:
                    exit_price = current_price
                    exit_time = current_time
                    exit_reason = 'time_exit'
                    break
                    
            if exit_price is None:
                return None
                
            # Apply spread and slippage to exit
            real_exit_price = self.apply_spread(exit_price, pair, 'short' if direction == 'long' else 'long')
            real_exit_price = self.apply_slippage(real_exit_price, pair, volatility)
            
            # Calculate final results
            if direction == 'long':
                pips = (real_exit_price - real_entry_price) / (0.01 if 'JPY' in pair else 0.0001)
            else:
                pips = (real_entry_price - real_exit_price) / (0.01 if 'JPY' in pair else 0.0001)
                
            pip_value = self.calculate_pip_value(pair, real_entry_price, position_size)
            profit_loss = pips * pip_value
            
            # Calculate costs
            commission = self.calculate_commission(position_size)
            spread_cost = self.config.spread_pips * pip_value * 2  # Entry and exit
            slippage_cost = abs(real_entry_price - entry_price) + abs(real_exit_price - exit_price)
            slippage_cost_pips = slippage_cost / (0.01 if 'JPY' in pair else 0.0001)
            
            # Risk-reward ratio
            if direction == 'long':
                potential_reward = (take_profit - real_entry_price) / (0.01 if 'JPY' in pair else 0.0001)
                potential_risk = (real_entry_price - stop_loss) / (0.01 if 'JPY' in pair else 0.0001)
            else:
                potential_reward = (real_entry_price - take_profit) / (0.01 if 'JPY' in pair else 0.0001)
                potential_risk = (stop_loss - real_entry_price) / (0.01 if 'JPY' in pair else 0.0001)
                
            risk_reward_ratio = potential_reward / potential_risk if potential_risk > 0 else 0
            
            holding_period = int((exit_time - entry_time).total_seconds() / 60)
            
            return TradeResult(
                entry_time=entry_time,
                exit_time=exit_time,
                pair=pair,
                direction=direction,
                entry_price=real_entry_price,
                exit_price=real_exit_price,
                position_size=position_size,
                pips=pips,
                profit_loss=profit_loss - commission - spread_cost,
                commission=commission,
                spread_cost=spread_cost,
                slippage_cost=slippage_cost_pips * pip_value,
                holding_period=holding_period,
                max_favorable_excursion=max_favorable_excursion,
                max_adverse_excursion=max_adverse_excursion,
                risk_reward_ratio=risk_reward_ratio,
                signal_confidence=confidence,
                market_regime=market_regime
            )
            
        except Exception as e:
            logger.error(f"Error simulating trade: {e}")
            return None
            
    def run_backtest(self, signals: List[Dict], market_data: Dict[str, List[Dict]], 
                    backtest_id: str = None) -> BacktestMetrics:
        """Run comprehensive backtest on historical signals"""
        if backtest_id is None:
            backtest_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        logger.info(f"Starting backtest {backtest_id} with {len(signals)} signals")
        
        self.trades = []
        balance = self.config.initial_balance
        self.equity_curve = [balance]
        
        for signal in signals:
            pair = signal['pair']
            
            # Get market data for this pair
            if pair not in market_data:
                logger.warning(f"No market data for {pair}, skipping signal")
                continue
                
            # Simulate trade
            trade_result = self.simulate_trade_execution(signal, market_data[pair])
            if trade_result:
                self.trades.append(trade_result)
                balance += trade_result.profit_loss
                self.equity_curve.append(balance)
                
                # Save trade to database
                self.save_trade_to_db(trade_result, backtest_id)
                
        # Calculate comprehensive metrics
        metrics = self.calculate_backtest_metrics()
        
        # Save backtest session
        self.save_backtest_session(backtest_id, metrics)
        
        logger.info(f"Backtest {backtest_id} completed. Total trades: {metrics.total_trades}, "
                   f"Win rate: {metrics.win_rate:.2%}, Total return: {metrics.total_return:.2%}")
        
        return metrics
        
    def calculate_backtest_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return BacktestMetrics(
                total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
                average_win=0.0, average_loss=0.0, profit_factor=0.0, total_return=0.0,
                max_drawdown=0.0, max_drawdown_duration=0, sharpe_ratio=0.0,
                sortino_ratio=0.0, calmar_ratio=0.0, expectancy=0.0, recovery_factor=0.0,
                consecutive_wins=0, consecutive_losses=0, avg_holding_period=0.0,
                avg_risk_reward=0.0, final_balance=self.config.initial_balance,
                total_profit=0.0, total_commission=0.0, total_spread_cost=0.0,
                total_slippage_cost=0.0, monthly_returns={}, yearly_returns={},
                signal_accuracy_by_confidence={}, performance_by_market_regime={},
                performance_by_hour={}, performance_by_day_of_week={}
            )
            
        # Basic trade statistics
        winning_trades = [t for t in self.trades if t.profit_loss > 0]
        losing_trades = [t for t in self.trades if t.profit_loss <= 0]
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        average_win = np.mean([t.profit_loss for t in winning_trades]) if winning_trades else 0
        average_loss = np.mean([t.profit_loss for t in losing_trades]) if losing_trades else 0
        
        total_profit = sum(t.profit_loss for t in self.trades)
        total_commission = sum(t.commission for t in self.trades)
        total_spread_cost = sum(t.spread_cost for t in self.trades)
        total_slippage_cost = sum(t.slippage_cost for t in self.trades)
        
        profit_factor = abs(average_win / average_loss) if average_loss != 0 else float('inf')
        
        # Return calculations
        final_balance = self.equity_curve[-1]
        total_return = (final_balance - self.config.initial_balance) / self.config.initial_balance
        
        # Drawdown calculations
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Find max drawdown duration
        max_drawdown_duration = 0
        current_duration = 0
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_drawdown_duration = max(max_drawdown_duration, current_duration)
            else:
                current_duration = 0
                
        # Sharpe ratio (assuming 0% risk-free rate)
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = [r for r in returns if r < 0]
        downside_std = np.std(downside_returns) if downside_returns else 0
        sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Expectancy
        expectancy = (win_rate * average_win) + ((1 - win_rate) * average_loss)
        
        # Recovery factor
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        current_streak = 0
        last_result = None
        
        for trade in self.trades:
            current_result = 'win' if trade.profit_loss > 0 else 'loss'
            
            if current_result == last_result:
                current_streak += 1
            else:
                current_streak = 1
                
            if current_result == 'win':
                consecutive_wins = max(consecutive_wins, current_streak)
            else:
                consecutive_losses = max(consecutive_losses, current_streak)
                
            last_result = current_result
            
        # Average statistics
        avg_holding_period = np.mean([t.holding_period for t in self.trades])
        avg_risk_reward = np.mean([t.risk_reward_ratio for t in self.trades])
        
        # Time-based analysis
        monthly_returns = self._calculate_time_based_returns('monthly')
        yearly_returns = self._calculate_time_based_returns('yearly')
        
        # Signal confidence analysis
        signal_accuracy_by_confidence = self._analyze_signal_confidence()
        
        # Market regime analysis
        performance_by_market_regime = self._analyze_market_regime_performance()
        
        # Hour and day of week analysis
        performance_by_hour = self._analyze_hourly_performance()
        performance_by_day_of_week = self._analyze_daily_performance()
        
        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            total_return=total_return,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            expectancy=expectancy,
            recovery_factor=recovery_factor,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            avg_holding_period=avg_holding_period,
            avg_risk_reward=avg_risk_reward,
            final_balance=final_balance,
            total_profit=total_profit,
            total_commission=total_commission,
            total_spread_cost=total_spread_cost,
            total_slippage_cost=total_slippage_cost,
            monthly_returns=monthly_returns,
            yearly_returns=yearly_returns,
            signal_accuracy_by_confidence=signal_accuracy_by_confidence,
            performance_by_market_regime=performance_by_market_regime,
            performance_by_hour=performance_by_hour,
            performance_by_day_of_week=performance_by_day_of_week
        )
        
    def _calculate_time_based_returns(self, period: str) -> Dict[str, float]:
        """Calculate returns by time period"""
        returns_by_period = {}
        
        for trade in self.trades:
            if period == 'monthly':
                key = trade.entry_time.strftime('%Y-%m')
            elif period == 'yearly':
                key = trade.entry_time.strftime('%Y')
            else:
                continue
                
            if key not in returns_by_period:
                returns_by_period[key] = []
            returns_by_period[key].append(trade.profit_loss)
            
        return {k: sum(v) for k, v in returns_by_period.items()}
        
    def _analyze_signal_confidence(self) -> Dict[str, float]:
        """Analyze performance by signal confidence levels"""
        confidence_levels = {'low': [], 'medium': [], 'high': []}
        
        for trade in self.trades:
            if trade.signal_confidence >= 0.8:
                confidence_levels['high'].append(trade.profit_loss)
            elif trade.signal_confidence >= 0.6:
                confidence_levels['medium'].append(trade.profit_loss)
            else:
                confidence_levels['low'].append(trade.profit_loss)
                
        return {
            'low': np.mean(confidence_levels['low']) if confidence_levels['low'] else 0,
            'medium': np.mean(confidence_levels['medium']) if confidence_levels['medium'] else 0,
            'high': np.mean(confidence_levels['high']) if confidence_levels['high'] else 0
        }
        
    def _analyze_market_regime_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by market regime"""
        regime_performance = {}
        
        for trade in self.trades:
            regime = trade.market_regime
            if regime not in regime_performance:
                regime_performance[regime] = {'trades': [], 'wins': 0}
                
            regime_performance[regime]['trades'].append(trade.profit_loss)
            if trade.profit_loss > 0:
                regime_performance[regime]['wins'] += 1
                
        result = {}
        for regime, data in regime_performance.items():
            trades = data['trades']
            result[regime] = {
                'avg_return': np.mean(trades) if trades else 0,
                'win_rate': data['wins'] / len(trades) if trades else 0,
                'total_trades': len(trades)
            }
            
        return result
        
    def _analyze_hourly_performance(self) -> Dict[int, float]:
        """Analyze performance by hour of day"""
        hourly_performance = {i: [] for i in range(24)}
        
        for trade in self.trades:
            hour = trade.entry_time.hour
            hourly_performance[hour].append(trade.profit_loss)
            
        return {hour: np.mean(profits) if profits else 0 
                for hour, profits in hourly_performance.items()}
                
    def _analyze_daily_performance(self) -> Dict[str, float]:
        """Analyze performance by day of week"""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_performance = {day: [] for day in days}
        
        for trade in self.trades:
            day = days[trade.entry_time.weekday()]
            daily_performance[day].append(trade.profit_loss)
            
        return {day: np.mean(profits) if profits else 0 
                for day, profits in daily_performance.items()}
                
    def save_trade_to_db(self, trade: TradeResult, backtest_id: str):
        """Save trade result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO backtest_trades (
                entry_time, exit_time, pair, direction, entry_price, exit_price,
                position_size, pips, profit_loss, commission, spread_cost,
                slippage_cost, holding_period, max_favorable_excursion,
                max_adverse_excursion, risk_reward_ratio, signal_confidence,
                market_regime, backtest_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.entry_time.isoformat(), trade.exit_time.isoformat(),
            trade.pair, trade.direction, trade.entry_price, trade.exit_price,
            trade.position_size, trade.pips, trade.profit_loss, trade.commission,
            trade.spread_cost, trade.slippage_cost, trade.holding_period,
            trade.max_favorable_excursion, trade.max_adverse_excursion,
            trade.risk_reward_ratio, trade.signal_confidence, trade.market_regime,
            backtest_id
        ))
        
        conn.commit()
        conn.close()
        
    def save_backtest_session(self, backtest_id: str, metrics: BacktestMetrics):
        """Save backtest session summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO backtest_sessions (
                id, start_time, end_time, initial_balance, final_balance,
                total_trades, win_rate, profit_factor, max_drawdown,
                sharpe_ratio, config
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            backtest_id,
            min(t.entry_time for t in self.trades).isoformat() if self.trades else datetime.now().isoformat(),
            max(t.exit_time for t in self.trades).isoformat() if self.trades else datetime.now().isoformat(),
            self.config.initial_balance,
            metrics.final_balance,
            metrics.total_trades,
            metrics.win_rate,
            metrics.profit_factor,
            metrics.max_drawdown,
            metrics.sharpe_ratio,
            json.dumps(self.config.__dict__)
        ))
        
        conn.commit()
        conn.close()
        
    def generate_report(self, metrics: BacktestMetrics) -> str:
        """Generate comprehensive backtest report"""
        report = f"""
# 📊 Advanced Backtesting Report

## Executive Summary
- **Total Trades**: {metrics.total_trades}
- **Win Rate**: {metrics.win_rate:.2%}
- **Total Return**: {metrics.total_return:.2%}
- **Max Drawdown**: {metrics.max_drawdown:.2%}
- **Sharpe Ratio**: {metrics.sharpe_ratio:.2f}
- **Final Balance**: ${metrics.final_balance:,.2f}

## Risk Metrics
- **Profit Factor**: {metrics.profit_factor:.2f}
- **Expectancy**: ${metrics.expectancy:.2f}
- **Sortino Ratio**: {metrics.sortino_ratio:.2f}
- **Calmar Ratio**: {metrics.calmar_ratio:.2f}
- **Recovery Factor**: {metrics.recovery_factor:.2f}

## Trade Statistics
- **Average Win**: ${metrics.average_win:.2f}
- **Average Loss**: ${metrics.average_loss:.2f}
- **Average Holding Period**: {metrics.avg_holding_period:.1f} minutes
- **Average Risk-Reward**: {metrics.avg_risk_reward:.2f}
- **Consecutive Wins**: {metrics.consecutive_wins}
- **Consecutive Losses**: {metrics.consecutive_losses}

## Cost Analysis
- **Total Commission**: ${metrics.total_commission:.2f}
- **Total Spread Cost**: ${metrics.total_spread_cost:.2f}
- **Total Slippage Cost**: ${metrics.total_slippage_cost:.2f}
- **Net Profit**: ${metrics.total_profit:.2f}

## Signal Quality Analysis
"""
        
        # Add signal confidence analysis
        if metrics.signal_accuracy_by_confidence:
            report += "\n### Signal Confidence Performance\n"
            for confidence, avg_return in metrics.signal_accuracy_by_confidence.items():
                report += f"- **{confidence.title()} Confidence**: ${avg_return:.2f} avg return\n"
                
        # Add market regime analysis
        if metrics.performance_by_market_regime:
            report += "\n### Market Regime Performance\n"
            for regime, data in metrics.performance_by_market_regime.items():
                report += f"- **{regime.title()}**: {data['win_rate']:.1%} win rate, "
                report += f"${data['avg_return']:.2f} avg return, {data['total_trades']} trades\n"
                
        # Add time-based analysis
        if metrics.performance_by_hour:
            best_hour = max(metrics.performance_by_hour.items(), key=lambda x: x[1])
            worst_hour = min(metrics.performance_by_hour.items(), key=lambda x: x[1])
            report += f"\n### Optimal Trading Times\n"
            report += f"- **Best Hour**: {best_hour[0]}:00 (${best_hour[1]:.2f} avg)\n"
            report += f"- **Worst Hour**: {worst_hour[0]}:00 (${worst_hour[1]:.2f} avg)\n"
            
        return report
        
    def get_historical_performance(self, days: int = 30) -> Dict[str, Any]:
        """Get recent backtest performance data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get trades from last N days
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT * FROM backtest_trades 
            WHERE entry_time >= ? 
            ORDER BY entry_time DESC
        ''', (cutoff_date,))
        
        trades = cursor.fetchall()
        conn.close()
        
        if not trades:
            return {'error': 'No recent trades found'}
            
        # Calculate recent performance metrics
        recent_trades = []
        for trade in trades:
            trade_dict = {
                'entry_time': trade[1],
                'exit_time': trade[2],
                'pair': trade[3],
                'direction': trade[4],
                'profit_loss': trade[9],
                'pips': trade[8],
                'confidence': trade[17],
                'market_regime': trade[18]
            }
            recent_trades.append(trade_dict)
            
        total_profit = sum(t['profit_loss'] for t in recent_trades)
        winning_trades = [t for t in recent_trades if t['profit_loss'] > 0]
        win_rate = len(winning_trades) / len(recent_trades) if recent_trades else 0
        
        return {
            'total_trades': len(recent_trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'average_profit': total_profit / len(recent_trades) if recent_trades else 0,
            'recent_trades': recent_trades[-10:]  # Last 10 trades
        }
