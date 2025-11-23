"""
Advanced Backtesting Engine with realistic trading conditions
Implements spread, commission, and slippage modeling
"""

import sqlite3
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger
import json

@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    direction: str  # 'long' or 'short'
    position_size: float
    entry_reason: str
    exit_reason: str
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    spread_cost: float = 0.0
    commission_cost: float = 0.0
    slippage_cost: float = 0.0
    total_costs: float = 0.0
    net_pnl: Optional[float] = None
    net_pnl_percentage: Optional[float] = None

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 10000.0
    max_risk_per_trade: float = 0.02  # 2% risk per trade
    max_daily_risk: float = 0.06  # 6% daily risk limit
    max_positions: int = 3  # Maximum concurrent positions
    spread_multiplier: float = 1.0  # Realistic spread multiplier
    commission_rate: float = 0.0001  # 0.01% commission per trade
    slippage_rate: float = 0.0002  # 0.02% slippage per trade
    min_sl_distance_atr: float = 1.5  # Minimum stop loss distance in ATR
    risk_reward_ratio: float = 2.0  # Target risk:reward ratio
    use_atr_position_sizing: bool = True
    use_dynamic_sl_tp: bool = True
    account_for_news: bool = True
    max_drawdown_limit: float = 0.20  # 20% maximum drawdown
    use_volatility_filter: bool = True
    min_win_probability: float = 0.55  # Minimum win probability for entry

class AdvancedBacktestingEngine:
    """Advanced backtesting engine with realistic trading conditions"""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.drawdown_curve: List[float] = []
        self.performance_metrics: Dict = {}
        self.db_path = "forex_bot.db"
        
    def calculate_realistic_costs(self, entry_price: float, exit_price: float, 
                                  position_size: float, pair: str, 
                                  market_conditions: Dict) -> Dict[str, float]:
        """Calculate realistic trading costs including spread, commission, and slippage"""
        
        # Get pair specifications
        pair_specs = self._get_pair_specs(pair)
        base_spread = pair_specs.get('spread', 0.0001)
        
        # Adjust spread based on market conditions
        volatility_multiplier = self._get_volatility_spread_multiplier(market_conditions)
        news_multiplier = self._get_news_spread_multiplier(market_conditions)
        time_multiplier = self._get_time_spread_multiplier(market_conditions)
        
        # Calculate effective spread
        effective_spread = base_spread * self.config.spread_multiplier * \
                          volatility_multiplier * news_multiplier * time_multiplier
        
        # Calculate costs
        spread_cost = effective_spread * position_size
        commission_cost = self.config.commission_rate * position_size * entry_price
        
        # Slippage calculation (higher during volatile periods)
        slippage_multiplier = volatility_multiplier * (1.0 + (news_multiplier - 1.0) * 0.5)
        slippage_cost = self.config.slippage_rate * position_size * entry_price * slippage_multiplier
        
        total_costs = spread_cost + commission_cost + slippage_cost
        
        return {
            'spread_cost': spread_cost,
            'commission_cost': commission_cost,
            'slippage_cost': slippage_cost,
            'total_costs': total_costs,
            'effective_spread': effective_spread
        }
    
    def _get_pair_specs(self, pair: str) -> Dict:
        """Get pair specifications with realistic trading parameters"""
        pair_specs = {
            'XAUUSD': {'spread': 0.2, 'pip_value': 0.01, 'tick_size': 0.01},
            'EURUSD': {'spread': 0.0001, 'pip_value': 0.0001, 'tick_size': 0.00001},
            'GBPUSD': {'spread': 0.00015, 'pip_value': 0.0001, 'tick_size': 0.00001},
            'USDJPY': {'spread': 0.015, 'pip_value': 0.01, 'tick_size': 0.001},
            'USDCHF': {'spread': 0.0002, 'pip_value': 0.0001, 'tick_size': 0.00001},
            'AUDUSD': {'spread': 0.00015, 'pip_value': 0.0001, 'tick_size': 0.00001},
            'USDCAD': {'spread': 0.00015, 'pip_value': 0.0001, 'tick_size': 0.00001},
            'NZDUSD': {'spread': 0.0002, 'pip_value': 0.0001, 'tick_size': 0.00001},
            'BTC': {'spread': 15.0, 'pip_value': 1.0, 'tick_size': 0.01},
            'AAPL': {'spread': 0.05, 'pip_value': 0.01, 'tick_size': 0.01},
            'USTEC': {'spread': 2.0, 'pip_value': 0.01, 'tick_size': 0.01},
            'USOIL': {'spread': 0.05, 'pip_value': 0.01, 'tick_size': 0.01}
        }
        return pair_specs.get(pair, {'spread': 0.0002, 'pip_value': 0.0001, 'tick_size': 0.00001})
    
    def _get_volatility_spread_multiplier(self, market_conditions: Dict) -> float:
        """Calculate spread multiplier based on volatility"""
        volatility = market_conditions.get('volatility', 0.01)
        base_volatility = 0.01  # 1% base volatility
        
        if volatility > base_volatility * 2:  # High volatility
            return 1.5 + (volatility - base_volatility * 2) / base_volatility * 0.5
        elif volatility < base_volatility * 0.5:  # Low volatility
            return 0.8
        else:  # Normal volatility
            return 1.0
    
    def _get_news_spread_multiplier(self, market_conditions: Dict) -> float:
        """Calculate spread multiplier based on news impact"""
        news_impact = market_conditions.get('news_impact', 0.0)
        
        if news_impact > 0.5:  # High news impact
            return 2.0
        elif news_impact > 0.2:  # Medium news impact
            return 1.3
        else:  # Low news impact
            return 1.0
    
    def _get_time_spread_multiplier(self, market_conditions: Dict) -> float:
        """Calculate spread multiplier based on trading session"""
        current_time = market_conditions.get('current_time', datetime.now())
        hour = current_time.hour
        
        # Asian session (low liquidity)
        if 0 <= hour < 8:
            return 1.2
        # European session (high liquidity)
        elif 8 <= hour < 16:
            return 0.9
        # American session (high liquidity)
        elif 16 <= hour < 24:
            return 1.0
        else:
            return 1.0
    
    def calculate_position_size(self, account_balance: float, entry_price: float, 
                               stop_loss_price: float, atr: float, pair: str,
                               market_conditions: Dict) -> float:
        """Calculate optimal position size based on risk management"""
        
        # Risk per trade in monetary terms
        risk_amount = account_balance * self.config.max_risk_per_trade
        
        # Calculate stop loss distance
        if self.config.use_atr_position_sizing and atr > 0:
            # Use ATR-based stop loss
            sl_distance = max(abs(entry_price - stop_loss_price), atr * self.config.min_sl_distance_atr)
        else:
            sl_distance = abs(entry_price - stop_loss_price)
        
        # Calculate position size based on risk
        if sl_distance > 0:
            position_size = risk_amount / sl_distance
        else:
            position_size = 0
        
        # Apply volatility filter
        if self.config.use_volatility_filter:
            volatility = market_conditions.get('volatility', 0.01)
            if volatility > 0.03:  # High volatility (>3%)
                position_size *= 0.7  # Reduce position size by 30%
            elif volatility < 0.005:  # Very low volatility (<0.5%)
                position_size *= 1.2  # Increase position size by 20%
        
        # Apply maximum position limit
        max_position_value = account_balance * 0.1  # Maximum 10% of account per position
        max_position_size = max_position_value / entry_price
        position_size = min(position_size, max_position_size)
        
        return max(0, position_size)  # Ensure non-negative
    
    def calculate_dynamic_sl_tp(self, entry_price: float, direction: str, atr: float,
                               market_regime: str, pair: str) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit levels based on market conditions"""
        
        if not self.config.use_dynamic_sl_tp or atr <= 0:
            # Use fixed risk:reward ratio
            if direction == 'long':
                stop_loss = entry_price * (1 - 0.01)  # 1% stop loss
                take_profit = entry_price * (1 + 0.02)  # 2% take profit
            else:  # short
                stop_loss = entry_price * (1 + 0.01)  # 1% stop loss
                take_profit = entry_price * (1 - 0.02)  # 2% take profit
            return stop_loss, take_profit
        
        # ATR-based dynamic SL/TP
        if market_regime == 'trending':
            sl_multiplier = 1.5  # Tighter stops in trending markets
            tp_multiplier = 3.0  # Larger targets in trending markets
        elif market_regime == 'ranging':
            sl_multiplier = 2.0  # Wider stops in ranging markets
            tp_multiplier = 2.0  # Smaller targets in ranging markets
        else:  # high_volatility
            sl_multiplier = 2.5  # Very wide stops in volatile markets
            tp_multiplier = 4.0  # Large targets to compensate for volatility
        
        if direction == 'long':
            stop_loss = entry_price - (atr * sl_multiplier)
            take_profit = entry_price + (atr * tp_multiplier)
        else:  # short
            stop_loss = entry_price + (atr * sl_multiplier)
            take_profit = entry_price - (atr * tp_multiplier)
        
        return stop_loss, take_profit
    
    def run_advanced_backtest(self, quotes: List[Dict], pair: str, 
                             ai_signals: List[Dict], market_conditions: List[Dict]) -> Dict:
        """Run advanced backtest with realistic trading conditions"""
        
        if len(quotes) < 100 or len(ai_signals) < 5:
            return self._empty_backtest_results()
        
        self.trades = []
        self.equity_curve = [self.config.initial_capital]
        self.drawdown_curve = [0.0]
        
        account_balance = self.config.initial_capital
        current_positions = []
        daily_pnl = []
        
        for i, quote in enumerate(quotes[50:], 50):  # Start from index 50 for indicators
            current_time = quote.get('timestamp', datetime.now())
            current_price = quote['close']
            
            # Get current market conditions
            current_conditions = market_conditions[i] if i < len(market_conditions) else {}
            current_conditions['current_time'] = current_time
            
            # Check for AI signals at this time
            current_signal = self._find_signal_at_time(ai_signals, current_time)
            
            if current_signal and len(current_positions) < self.config.max_positions:
                # Check win probability
                if current_signal.get('win_probability', 0.5) >= self.config.min_win_probability:
                    # Calculate position parameters
                    direction = current_signal.get('direction', 'long')
                    atr = current_conditions.get('atr', current_price * 0.01)
                    market_regime = current_conditions.get('market_regime', 'ranging')
                    
                    # Calculate dynamic SL/TP
                    entry_price = current_price
                    stop_loss, take_profit = self.calculate_dynamic_sl_tp(
                        entry_price, direction, atr, market_regime, pair
                    )
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(
                        account_balance, entry_price, stop_loss, atr, pair, current_conditions
                    )
                    
                    if position_size > 0:
                        # Calculate realistic trading costs
                        costs = self.calculate_realistic_costs(
                            entry_price, entry_price, position_size, pair, current_conditions
                        )
                        
                        # Create new position
                        trade = Trade(
                            entry_time=current_time,
                            exit_time=None,
                            entry_price=entry_price,
                            exit_price=None,
                            direction=direction,
                            position_size=position_size,
                            entry_reason=current_signal.get('reason', 'AI Signal'),
                            exit_reason='',
                            spread_cost=costs['spread_cost'],
                            commission_cost=costs['commission_cost'],
                            slippage_cost=costs['slippage_cost'],
                            total_costs=costs['total_costs']
                        )
                        
                        current_positions.append(trade)
                        
                        # Deduct costs from account
                        account_balance -= costs['total_costs']
            
            # Check for exit conditions
            for trade in current_positions[:]:
                exit_triggered = False
                exit_reason = ''
                exit_price = current_price
                
                # Stop loss check
                if trade.direction == 'long' and current_price <= stop_loss:
                    exit_triggered = True
                    exit_reason = 'Stop Loss'
                elif trade.direction == 'short' and current_price >= stop_loss:
                    exit_triggered = True
                    exit_reason = 'Stop Loss'
                
                # Take profit check
                elif trade.direction == 'long' and current_price >= take_profit:
                    exit_triggered = True
                    exit_reason = 'Take Profit'
                elif trade.direction == 'short' and current_price <= take_profit:
                    exit_triggered = True
                    exit_reason = 'Take Profit'
                
                # Time-based exit (after 24 hours)
                elif (current_time - trade.entry_time).total_seconds() > 86400:
                    exit_triggered = True
                    exit_reason = 'Time Exit'
                
                # Drawdown limit check
                elif account_balance < self.config.initial_capital * (1 - self.config.max_drawdown_limit):
                    exit_triggered = True
                    exit_reason = 'Drawdown Limit'
                
                if exit_triggered:
                    # Calculate exit costs
                    exit_costs = self.calculate_realistic_costs(
                        exit_price, exit_price, trade.position_size, pair, current_conditions
                    )
                    
                    # Calculate P&L
                    if trade.direction == 'long':
                        gross_pnl = (exit_price - trade.entry_price) * trade.position_size
                    else:  # short
                        gross_pnl = (trade.entry_price - exit_price) * trade.position_size
                    
                    net_pnl = gross_pnl - exit_costs['total_costs'] - trade.total_costs
                    net_pnl_percentage = (net_pnl / (trade.entry_price * trade.position_size)) * 100
                    
                    # Update trade
                    trade.exit_time = current_time
                    trade.exit_price = exit_price
                    trade.exit_reason = exit_reason
                    trade.pnl = gross_pnl
                    trade.net_pnl = net_pnl
                    trade.net_pnl_percentage = net_pnl_percentage
                    
                    # Update account balance
                    account_balance += net_pnl
                    
                    # Remove from current positions
                    current_positions.remove(trade)
                    self.trades.append(trade)
            
            # Update equity curve
            self.equity_curve.append(account_balance)
            
            # Calculate drawdown
            peak = max(self.equity_curve)
            current_drawdown = (peak - account_balance) / peak if peak > 0 else 0
            self.drawdown_curve.append(current_drawdown)
            
            # Check daily risk limit
            if len(daily_pnl) > 0 and (current_time - daily_pnl[-1]['date']).days >= 1:
                daily_pnl = []  # Reset daily P&L
            
            # Check if daily risk limit exceeded
            if len(daily_pnl) > 0:
                daily_loss = sum(p['pnl'] for p in daily_pnl if p['pnl'] < 0)
                daily_risk = abs(daily_loss) / self.config.initial_capital
                if daily_risk > self.config.max_daily_risk:
                    # Close all positions
                    for trade in current_positions[:]:
                        self._force_exit_trade(trade, current_price, current_time, 'Daily Risk Limit')
            
            # Add to daily P&L
            if len(self.trades) > 0 and self.trades[-1].exit_time == current_time:
                daily_pnl.append({'date': current_time, 'pnl': self.trades[-1].net_pnl})
        
        # Force exit any remaining positions
        for trade in current_positions[:]:
            self._force_exit_trade(trade, quotes[-1]['close'], quotes[-1].get('timestamp', datetime.now()), 'End of Data')
        
        # Calculate performance metrics
        return self._calculate_performance_metrics()
    
    def _find_signal_at_time(self, ai_signals: List[Dict], target_time: datetime) -> Optional[Dict]:
        """Find AI signal at specific time"""
        for signal in ai_signals:
            signal_time = signal.get('timestamp', datetime.now())
            if abs((signal_time - target_time).total_seconds()) < 3600:  # Within 1 hour
                return signal
        return None
    
    def _force_exit_trade(self, trade: Trade, current_price: float, current_time: datetime, reason: str):
        """Force exit a trade"""
        trade.exit_time = current_time
        trade.exit_price = current_price
        trade.exit_reason = reason
        
        # Calculate P&L
        if trade.direction == 'long':
            gross_pnl = (current_price - trade.entry_price) * trade.position_size
        else:  # short
            gross_pnl = (trade.entry_price - current_price) * trade.position_size
        
        net_pnl = gross_pnl - trade.total_costs
        trade.pnl = gross_pnl
        trade.net_pnl = net_pnl
        trade.net_pnl_percentage = (net_pnl / (trade.entry_price * trade.position_size)) * 100
        
        self.trades.append(trade)
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return self._empty_backtest_results()
        
        # Basic statistics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.net_pnl > 0]
        losing_trades = [t for t in self.trades if t.net_pnl <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # P&L statistics
        total_gross_pnl = sum(t.pnl for t in self.trades)
        total_net_pnl = sum(t.net_pnl for t in self.trades)
        total_costs = sum(t.total_costs for t in self.trades)
        
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        
        net_profit = sum(t.net_pnl for t in winning_trades)
        net_loss = abs(sum(t.net_pnl for t in losing_trades))
        
        # Performance ratios
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        net_profit_factor = net_profit / net_loss if net_loss > 0 else float('inf')
        
        # Average trade statistics
        avg_win = np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.net_pnl for t in losing_trades]) if losing_trades else 0
        avg_trade = np.mean([t.net_pnl for t in self.trades])
        
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Return statistics
        total_return = (self.equity_curve[-1] - self.config.initial_capital) / self.config.initial_capital
        annualized_return = self._calculate_annualized_return(total_return)
        
        # Risk statistics
        max_drawdown = max(self.drawdown_curve) if self.drawdown_curve else 0
        avg_drawdown = np.mean(self.drawdown_curve) if self.drawdown_curve else 0
        
        # Sharpe ratio calculation
        returns = [self.equity_curve[i] / self.equity_curve[i-1] - 1 
                  for i in range(1, len(self.equity_curve))]
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        
        # Trade duration statistics
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 
                    for t in self.trades if t.exit_time and t.entry_time]
        avg_duration = np.mean(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        
        # Cost analysis
        avg_spread_cost = np.mean([t.spread_cost for t in self.trades])
        avg_commission_cost = np.mean([t.commission_cost for t in self.trades])
        avg_slippage_cost = np.mean([t.slippage_cost for t in self.trades])
        avg_total_cost = np.mean([t.total_costs for t in self.trades])
        
        cost_impact = (total_costs / abs(total_gross_pnl)) * 100 if total_gross_pnl != 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': total_net_pnl,
            'total_costs': total_costs,
            'profit_factor': profit_factor,
            'net_profit_factor': net_profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'win_loss_ratio': win_loss_ratio,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'equity_curve': self.equity_curve,
            'drawdown_curve': self.drawdown_curve,
            'trades': self.trades,
            'avg_duration_hours': avg_duration,
            'max_duration_hours': max_duration,
            'min_duration_hours': min_duration,
            'cost_analysis': {
                'avg_spread_cost': avg_spread_cost,
                'avg_commission_cost': avg_commission_cost,
                'avg_slippage_cost': avg_slippage_cost,
                'avg_total_cost': avg_total_cost,
                'cost_impact_percentage': cost_impact
            },
            'final_equity': self.equity_curve[-1] if self.equity_curve else self.config.initial_capital,
            'initial_capital': self.config.initial_capital
        }
    
    def _calculate_annualized_return(self, total_return: float) -> float:
        """Calculate annualized return"""
        if not self.trades:
            return 0.0
        
        # Calculate trading period in years
        start_time = min(t.entry_time for t in self.trades)
        end_time = max(t.exit_time for t in self.trades if t.exit_time)
        
        if not end_time or start_time == end_time:
            return 0.0
        
        years = (end_time - start_time).total_seconds() / (365.25 * 24 * 3600)
        if years <= 0:
            return 0.0
        
        return ((1 + total_return) ** (1 / years)) - 1
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        excess_returns = [r - risk_free_rate / 252 for r in returns]  # Daily risk-free rate
        avg_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns)
        
        if std_excess_return == 0:
            return 0.0
        
        return avg_excess_return / std_excess_return * np.sqrt(252)  # Annualized
    
    def _empty_backtest_results(self) -> Dict:
        """Return empty backtest results"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'net_profit': 0.0,
            'total_costs': 0.0,
            'profit_factor': 0.0,
            'net_profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_trade': 0.0,
            'win_loss_ratio': 0.0,
            'total_return': 0.0,
            'annualized_return': 0.0,
            'max_drawdown': 0.0,
            'avg_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'equity_curve': [self.config.initial_capital],
            'drawdown_curve': [0.0],
            'trades': [],
            'avg_duration_hours': 0.0,
            'max_duration_hours': 0.0,
            'min_duration_hours': 0.0,
            'cost_analysis': {
                'avg_spread_cost': 0.0,
                'avg_commission_cost': 0.0,
                'avg_slippage_cost': 0.0,
                'avg_total_cost': 0.0,
                'cost_impact_percentage': 0.0
            },
            'final_equity': self.config.initial_capital,
            'initial_capital': self.config.initial_capital
        }
    
    def save_backtest_results(self, results: Dict, pair: str, timeframe: str, 
                            ai_model: str, notes: str = "") -> bool:
        """Save backtest results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO backtest_results (
                    pair, timeframe, start_date, end_date, total_trades, win_rate,
                    profit_factor, max_drawdown, total_return, ai_model, notes,
                    gross_profit, gross_loss, net_profit, total_costs, sharpe_ratio,
                    avg_win, avg_loss, win_loss_ratio, avg_duration_hours
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pair,
                timeframe,
                min(t.entry_time for t in results['trades']) if results['trades'] else datetime.now(),
                max(t.exit_time for t in results['trades'] if t.exit_time) if results['trades'] else datetime.now(),
                results['total_trades'],
                results['win_rate'],
                results['profit_factor'],
                results['max_drawdown'],
                results['total_return'],
                ai_model,
                notes,
                results['gross_profit'],
                results['gross_loss'],
                results['net_profit'],
                results['total_costs'],
                results['sharpe_ratio'],
                results['avg_win'],
                results['avg_loss'],
                results['win_loss_ratio'],
                results['avg_duration_hours']
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Backtest results saved for {pair} {timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
            return False
    
    def generate_backtest_report(self, results: Dict, pair: str, timeframe: str) -> str:
        """Generate comprehensive backtest report"""
        if results['total_trades'] == 0:
            return f"📊 *Backtest Report: {pair} {timeframe}*\n\n❌ Нет сделок для анализа"
        
        report = f"""
📊 *Advanced Backtest Report: {pair} {timeframe}*

📈 *Основные метрики:*
• Всего сделок: {results['total_trades']}
• Прибыльных: {results['winning_trades']} ({results['win_rate']:.1%})
• Убыточных: {results['losing_trades']}
• Точность: {results['win_rate']:.1%} {'🟢' if results['win_rate'] >= 0.6 else '🟡' if results['win_rate'] >= 0.4 else '🔴'}

💰 *Финансовые результаты:*
• Валовая прибыль: ${results['gross_profit']:.2f}
• Валовый убыток: ${results['gross_loss']:.2f}
• Чистая прибыль: ${results['net_profit']:.2f} {'🟢' if results['net_profit'] > 0 else '🔴'}
• Общие издержки: ${results['total_costs']:.2f}
• Итоговая доходность: {results['total_return']:.2%}
• Годовая доходность: {results['annualized_return']:.2%}

📊 *Коэффициенты:*
• Профит-фактор: {results['profit_factor']:.2f} {'🟢' if results['profit_factor'] > 1.5 else '🟡' if results['profit_factor'] > 1.0 else '🔴'}
• Чистый профит-фактор: {results['net_profit_factor']:.2f}
• Соотношение прибыль/убыток: {results['win_loss_ratio']:.2f}:1
• Коэффициент Шарпа: {results['sharpe_ratio']:.2f} {'🟢' if results['sharpe_ratio'] > 1.0 else '🟡' if results['sharpe_ratio'] > 0.5 else '🔴'}

📉 *Риски:*
• Макс. просадка: {results['max_drawdown']:.2%} {'🟢' if results['max_drawdown'] < 0.2 else '🟡' if results['max_drawdown'] < 0.3 else '🔴'}
• Средняя просадка: {results['avg_drawdown']:.2%}
• Средняя сделка: ${results['avg_trade']:.2f}
• Средний выигрыш: ${results['avg_win']:.2f}
• Средний проигрыш: ${results['avg_loss']:.2f}

⏱️ *Временные характеристики:*
• Средняя длительность: {results['avg_duration_hours']:.1f}ч
• Максимальная длительность: {results['max_duration_hours']:.1f}ч
• Минимальная длительность: {results['min_duration_hours']:.1f}ч

💸 *Анализ издержек:*
• Средний спред: ${results['cost_analysis']['avg_spread_cost']:.2f}
• Средняя комиссия: ${results['cost_analysis']['avg_commission_cost']:.2f}
• Средний проскальзывание: ${results['cost_analysis']['avg_slippage_cost']:.2f}
• Средние общие издержки: ${results['cost_analysis']['avg_total_cost']:.2f}
• Влияние издержек: {results['cost_analysis']['cost_impact_percentage']:.1f}%

📊 *Итоговая оценка:*
"""
        
        # Overall assessment
        score = 0
        if results['win_rate'] >= 0.6: score += 2
        elif results['win_rate'] >= 0.4: score += 1
        
        if results['profit_factor'] >= 1.5: score += 2
        elif results['profit_factor'] >= 1.0: score += 1
        
        if results['sharpe_ratio'] >= 1.0: score += 2
        elif results['sharpe_ratio'] >= 0.5: score += 1
        
        if results['max_drawdown'] < 0.2: score += 2
        elif results['max_drawdown'] < 0.3: score += 1
        
        if results['total_return'] > 0: score += 2
        elif results['total_return'] > -0.1: score += 1
        
        if score >= 8:
            assessment = "🟢 ОТЛИЧНО - Высокоприбыльная стратегия"
        elif score >= 6:
            assessment = "🟡 ХОРОШО - Прибыльная стратегия с потенциалом"
        elif score >= 4:
            assessment = "🟠 УДОВЛЕТВОРИТЕЛЬНО - Средняя стратегия"
        else:
            assessment = "🔴 ПЛОХО - Стратегия требует доработки"
        
        report += f"• Общая оценка: {assessment}\n"
        report += f"• Балл: {score}/10\n"
        
        report += f"""

⚠️ *Важно:*
• Результаты основаны на исторических данных
• Прошлые результаты не гарантируют будущую прибыльность
• Всегда тестируйте стратегии на демо-счете
• Используйте адекватное управление рисками
"""
        
        return report