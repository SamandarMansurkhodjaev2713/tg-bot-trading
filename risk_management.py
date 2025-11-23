import math
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    """Risk levels for position sizing"""
    VERY_LOW = 0.005  # 0.5%
    LOW = 0.01        # 1%
    MEDIUM = 0.02     # 2%
    HIGH = 0.03       # 3%
    VERY_HIGH = 0.05  # 5%

class MarketRegime(Enum):
    """Market regime classifications"""
    TREND = "trend"
    RANGE = "range"
    HIGH_VOLATILITY = "high_volatility"
    UNKNOWN = "unknown"

@dataclass
class PositionSizingParams:
    """Parameters for position sizing calculation"""
    account_balance: float
    entry_price: float
    stop_loss_price: float
    risk_percentage: float
    atr: float
    market_regime: str
    signal_confidence: float
    pair: str
    time_to_event: Optional[float] = None  # hours to next high-impact event
    upcoming_events: Optional[List[Dict]] = None  # List of upcoming economic events

@dataclass
class RiskMetrics:
    """Risk metrics for a position"""
    position_size: float
    risk_amount: float
    risk_percentage: float
    reward_to_risk_ratio: float
    expected_value: float
    position_value: float
    leverage: float
    max_drawdown_estimate: float

class RiskManager:
    """Advanced risk management system for AI trading"""
    
    def __init__(self, db_path: str = "risk_management.db"):
        self.db_path = db_path
        self.risk_limits = {
            'max_risk_per_trade': 0.02,      # 2% max risk per trade
            'max_daily_risk': 0.05,          # 5% max daily risk
            'max_monthly_risk': 0.10,        # 10% max monthly risk
            'max_positions': 10,              # Max concurrent positions
            'max_correlated_positions': 3,    # Max correlated positions
            'min_account_balance': 1000,      # Minimum account balance
            'max_leverage': 10                # Maximum leverage
        }
        
        # Market regime risk multipliers
        self.regime_multipliers = {
            MarketRegime.TREND.value: 1.0,
            MarketRegime.RANGE.value: 0.8,
            MarketRegime.HIGH_VOLATILITY.value: 0.6,
            MarketRegime.UNKNOWN.value: 0.5
        }
        
        # Pair-specific risk adjustments
        self.pair_risk_adjustments = {
            'XAUUSD': 0.8,     # Gold is less volatile
            'XAGUSD': 1.2,     # Silver is more volatile
            'BTCUSD': 1.5,     # Bitcoin is highly volatile
            'ETHUSD': 1.3,     # Ethereum is volatile
            'US30': 0.9,       # Dow Jones
            'US100': 1.1,      # NASDAQ is more volatile
            'US500': 1.0,      # S&P 500
            'USTEC': 1.1,      # NASDAQ
            'USOIL': 1.4,      # Oil is volatile
            'EURUSD': 0.9,     # EUR/USD is relatively stable
            'GBPUSD': 1.1,     # GBP/USD is more volatile
            'USDJPY': 0.8,     # USD/JPY is less volatile
            'AUDUSD': 1.2,     # AUD/USD is volatile
            'NZDUSD': 1.3,     # NZD/USD is very volatile
        }
        
        # Time-based risk adjustments
        self.time_risk_adjustments = {
            'weekend': 0.5,    # Reduce risk before weekends
            'holiday': 0.3,    # Reduce risk before holidays
            'news_event': 0.3, # Reduce risk before major news
            'market_close': 0.4, # Reduce risk before market close
            'market_open': 0.8  # Slightly reduce risk after market open
        }
        
        # High-impact news events that should trigger trading avoidance
        self.high_impact_events = [
            'Non Farm Payrolls', 'Federal Funds Rate', 'GDP', 'CPI', 'PPI',
            'Unemployment Rate', 'ECB Rate Decision', 'BoE Rate Decision',
            'Fed Chairman Speech', 'ECB President Speech', 'BoJ Rate Decision',
            'NFP', 'FOMC', 'Fed Chair', 'Interest Rate', 'Inflation Rate',
            'Employment Change', 'Retail Sales', 'ISM Manufacturing',
            'Consumer Confidence', 'Home Sales', 'Trade Balance'
        ]
        
        self.init_database()
    
    def init_database(self):
        """Initialize risk management database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Risk events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                description TEXT,
                risk_amount REAL,
                account_balance REAL,
                pair TEXT,
                outcome TEXT,
                metadata TEXT
            )
        ''')
        
        # Position tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                pair TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                position_size REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL,
                risk_amount REAL NOT NULL,
                risk_percentage REAL NOT NULL,
                account_balance REAL NOT NULL,
                market_regime TEXT,
                signal_confidence REAL,
                status TEXT DEFAULT 'open',
                exit_price REAL,
                exit_timestamp DATETIME,
                pnl REAL,
                outcome TEXT,
                metadata TEXT
            )
        ''')
        
        # Daily risk tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_risk (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                total_risk REAL NOT NULL,
                total_pnl REAL,
                trades_count INTEGER,
                win_rate REAL,
                max_drawdown REAL,
                UNIQUE(date)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def calculate_position_size(self, params: PositionSizingParams) -> RiskMetrics:
        """Calculate optimal position size based on risk parameters"""
        try:
            # Check for news event avoidance first
            news_avoidance = self.check_news_event_avoidance(params.pair, params.upcoming_events or [])
            if news_avoidance['should_avoid']:
                # Return zero position size with avoidance reason
                return RiskMetrics(
                    position_size=0.0,
                    risk_amount=0.0,
                    risk_percentage=0.0,
                    reward_to_risk_ratio=0.0,
                    expected_value=0.0,
                    position_value=0.0,
                    leverage=0.0,
                    max_drawdown_estimate=0.0
                )
            
            # Base risk calculation
            base_risk_amount = params.account_balance * params.risk_percentage
            
            # Apply market regime multiplier
            regime_multiplier = self.regime_multipliers.get(
                params.market_regime, self.regime_multipliers[MarketRegime.UNKNOWN.value]
            )
            
            # Apply pair-specific adjustment
            pair_adjustment = self.pair_risk_adjustments.get(params.pair, 1.0)
            
            # Apply confidence-based adjustment
            confidence_adjustment = self._get_confidence_adjustment(params.signal_confidence)
            
            # Apply time-based adjustments (news events, etc.)
            time_adjustment = self._get_time_adjustment(params)
            
            # Calculate adjusted risk amount
            adjusted_risk_amount = base_risk_amount * regime_multiplier * pair_adjustment * confidence_adjustment * time_adjustment
            
            # Calculate risk per unit (distance to stop loss)
            risk_per_unit = abs(params.entry_price - params.stop_loss_price)
            if risk_per_unit == 0:
                risk_per_unit = params.atr * 0.5  # Fallback to half ATR
            
            # Calculate position size
            position_size = adjusted_risk_amount / risk_per_unit
            
            # Apply maximum position size limits
            max_position_size = self._get_max_position_size(params)
            position_size = min(position_size, max_position_size)
            
            # Calculate additional metrics
            position_value = position_size * params.entry_price
            leverage = position_value / params.account_balance
            
            # Calculate expected value (simplified)
            expected_value = self._calculate_expected_value(params, position_size)
            
            # Calculate reward-to-risk ratio
            # Assume 2:1 reward-to-risk as default
            take_profit_distance = risk_per_unit * 2
            reward_to_risk_ratio = take_profit_distance / risk_per_unit if risk_per_unit > 0 else 2.0
            
            # Estimate maximum drawdown
            max_drawdown_estimate = self._estimate_max_drawdown(params, position_size)
            
            return RiskMetrics(
                position_size=position_size,
                risk_amount=adjusted_risk_amount,
                risk_percentage=adjusted_risk_amount / params.account_balance,
                reward_to_risk_ratio=reward_to_risk_ratio,
                expected_value=expected_value,
                position_value=position_value,
                leverage=leverage,
                max_drawdown_estimate=max_drawdown_estimate
            )
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            # Return conservative position size
            return RiskMetrics(
                position_size=params.account_balance * 0.01 / params.atr,
                risk_amount=params.account_balance * 0.01,
                risk_percentage=0.01,
                reward_to_risk_ratio=2.0,
                expected_value=0.0,
                position_value=params.account_balance * 0.01 / params.atr * params.entry_price,
                leverage=0.01 / params.atr * params.entry_price,
                max_drawdown_estimate=0.02
            )
    
    def _get_confidence_adjustment(self, confidence: float) -> float:
        """Get risk adjustment based on signal confidence"""
        if confidence >= 0.8:
            return 1.2  # Increase risk for high confidence
        elif confidence >= 0.6:
            return 1.0  # Normal risk
        elif confidence >= 0.4:
            return 0.8  # Reduce risk for low confidence
        else:
            return 0.5  # Significantly reduce risk for very low confidence
    
    def _get_time_adjustment(self, params: PositionSizingParams) -> float:
        """Get risk adjustment based on time factors"""
        adjustment = 1.0
        
        # Check for news events
        if params.time_to_event is not None and params.time_to_event < 2:  # Less than 2 hours
            adjustment *= self.time_risk_adjustments['news_event']
        
        # Weekend adjustment (Friday after 4 PM)
        current_time = datetime.now()
        if current_time.weekday() == 4 and current_time.hour >= 16:  # Friday after 4 PM
            adjustment *= self.time_risk_adjustments['weekend']
        
        # Market close adjustment
        if current_time.hour >= 16:  # After 4 PM
            adjustment *= self.time_risk_adjustments['market_close']
        
        # Market open adjustment
        if current_time.hour < 10:  # Before 10 AM
            adjustment *= self.time_risk_adjustments['market_open']
        
        return max(0.1, adjustment)  # Minimum 10% of normal risk
    
    def _get_max_position_size(self, params: PositionSizingParams) -> float:
        """Get maximum allowed position size"""
        # Maximum position based on account balance and leverage
        max_position_value = params.account_balance * self.risk_limits['max_leverage']
        max_position_size = max_position_value / params.entry_price
        
        # Maximum position based on risk per trade
        max_risk_amount = params.account_balance * self.risk_limits['max_risk_per_trade']
        risk_per_unit = abs(params.entry_price - params.stop_loss_price)
        if risk_per_unit == 0:
            risk_per_unit = params.atr * 0.5
        
        max_position_by_risk = max_risk_amount / risk_per_unit
        
        # Return the minimum of both limits
        return min(max_position_size, max_position_by_risk)
    
    def _calculate_expected_value(self, params: PositionSizingParams, position_size: float) -> float:
        """Calculate expected value of the trade"""
        try:
            # Risk per unit
            risk_per_unit = abs(params.entry_price - params.stop_loss_price)
            if risk_per_unit == 0:
                risk_per_unit = params.atr * 0.5
            
            # Assume 2:1 reward-to-risk ratio
            reward_per_unit = risk_per_unit * 2
            
            # Total risk and reward
            total_risk = position_size * risk_per_unit
            total_reward = position_size * reward_per_unit
            
            # Win probability based on signal confidence
            win_probability = params.signal_confidence
            loss_probability = 1 - win_probability
            
            # Expected value
            expected_value = (win_probability * total_reward) - (loss_probability * total_risk)
            
            return expected_value
            
        except Exception as e:
            print(f"Error calculating expected value: {e}")
            return 0.0
    
    def _estimate_max_drawdown(self, params: PositionSizingParams, position_size: float) -> float:
        """Estimate maximum drawdown for the position"""
        try:
            # Base drawdown estimate (2x the risk amount)
            base_drawdown = position_size * abs(params.entry_price - params.stop_loss_price) * 2
            
            # Apply market regime multiplier
            regime_multiplier = self.regime_multipliers.get(
                params.market_regime, self.regime_multipliers[MarketRegime.UNKNOWN.value]
            )
            
            # Apply pair-specific adjustment
            pair_adjustment = self.pair_risk_adjustments.get(params.pair, 1.0)
            
            # Calculate estimated drawdown
            estimated_drawdown = base_drawdown * (2 - regime_multiplier) * pair_adjustment
            
            # Return as percentage of account balance
            return estimated_drawdown / params.account_balance
            
        except Exception as e:
            print(f"Error estimating max drawdown: {e}")
            return 0.02  # Default 2% drawdown estimate
    
    def check_risk_limits(self, account_balance: float, current_positions: List[Dict]) -> Dict[str, bool]:
        """Check if current risk exposure is within limits"""
        try:
            results = {}
            
            # Calculate current risk exposure
            total_risk = sum(pos.get('risk_amount', 0) for pos in current_positions)
            daily_risk = self._get_daily_risk_exposure()
            monthly_risk = self._get_monthly_risk_exposure()
            
            # Check individual limits
            results['within_max_risk_per_trade'] = all(
                pos.get('risk_percentage', 0) <= self.risk_limits['max_risk_per_trade']
                for pos in current_positions
            )
            
            results['within_daily_risk_limit'] = daily_risk <= self.risk_limits['max_daily_risk']
            results['within_monthly_risk_limit'] = monthly_risk <= self.risk_limits['max_monthly_risk']
            results['within_position_count_limit'] = len(current_positions) <= self.risk_limits['max_positions']
            
            # Check correlated positions
            results['within_correlated_positions_limit'] = self._check_correlated_positions_limit(current_positions)
            
            # Check minimum account balance
            results['above_minimum_balance'] = account_balance >= self.risk_limits['min_account_balance']
            
            # Check leverage limits
            results['within_leverage_limits'] = self._check_leverage_limits(current_positions, account_balance)
            
            return results
            
        except Exception as e:
            print(f"Error checking risk limits: {e}")
            return {key: False for key in [
                'within_max_risk_per_trade', 'within_daily_risk_limit', 
                'within_monthly_risk_limit', 'within_position_count_limit',
                'within_correlated_positions_limit', 'above_minimum_balance',
                'within_leverage_limits'
            ]}
    
    def _get_daily_risk_exposure(self) -> float:
        """Get current daily risk exposure"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = datetime.now().date()
            cursor.execute('''
                SELECT total_risk FROM daily_risk 
                WHERE date = ?
            ''', (today,))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else 0.0
            
        except Exception as e:
            print(f"Error getting daily risk exposure: {e}")
            return 0.0
    
    def _get_monthly_risk_exposure(self) -> float:
        """Get current monthly risk exposure"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get first day of current month
            first_day = datetime.now().replace(day=1).date()
            
            cursor.execute('''
                SELECT SUM(total_risk) FROM daily_risk 
                WHERE date >= ?
            ''', (first_day,))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result and result[0] else 0.0
            
        except Exception as e:
            print(f"Error getting monthly risk exposure: {e}")
            return 0.0
    
    def check_news_event_avoidance(self, pair: str, upcoming_events: List[Dict]) -> Dict[str, Any]:
        """Check if trading should be avoided due to upcoming news events"""
        try:
            result = {
                'should_avoid': False,
                'reason': '',
                'time_to_event': None,
                'event_impact': 'low',
                'events_affecting': []
            }
            
            if not upcoming_events:
                return result
            
            # Filter events that affect the trading pair
            affecting_events = []
            current_time = datetime.now()
            
            for event in upcoming_events:
                try:
                    # Parse event time and make it timezone-naive for comparison
                    event_timestamp = event['timestamp'].replace('Z', '+00:00')
                    event_time = datetime.fromisoformat(event_timestamp)
                    if event_time.tzinfo is not None:
                        event_time = event_time.replace(tzinfo=None)
                    
                    time_to_event = (event_time - current_time).total_seconds() / 3600  # hours
                    
                    # Only consider events within next 24 hours
                    if 0 < time_to_event <= 24:
                        event_name = event.get('event', '').lower()
                        event_impact = event.get('importance', 'low').lower()
                        
                        # Check if this event affects the trading pair
                        if self._event_affects_pair(event_name, pair):
                            affecting_events.append({
                                'event': event['event'],
                                'time_to_event': time_to_event,
                                'impact': event_impact,
                                'country': event.get('country', ''),
                                'actual': event.get('actual', ''),
                                'forecast': event.get('forecast', ''),
                                'previous': event.get('previous', '')
                            })
                except Exception as e:
                    continue
            
            if not affecting_events:
                return result
            
            # Sort by time to event
            affecting_events.sort(key=lambda x: x['time_to_event'])
            
            # Find the most impactful event within critical time window
            critical_event = None
            for event in affecting_events:
                if event['impact'] in ['high', 'medium'] and event['time_to_event'] <= 2:  # Within 2 hours
                    critical_event = event
                    break
                elif event['impact'] == 'high' and event['time_to_event'] <= 6:  # High impact within 6 hours
                    critical_event = event
                    break
            
            if critical_event:
                result['should_avoid'] = True
                result['reason'] = f"High-impact event '{critical_event['event']}' in {critical_event['time_to_event']:.1f} hours"
                result['time_to_event'] = critical_event['time_to_event']
                result['event_impact'] = critical_event['impact']
                result['events_affecting'] = affecting_events
            
            return result
            
        except Exception as e:
            print(f"Error checking news event avoidance: {e}")
            return {'should_avoid': False, 'reason': 'Error checking events', 'time_to_event': None, 'event_impact': 'low', 'events_affecting': []}
    
    def _event_affects_pair(self, event_name: str, pair: str) -> bool:
        """Check if an economic event affects the trading pair"""
        try:
            event_lower = event_name.lower()
            
            # Check if it's a high-impact event
            for high_impact_event in self.high_impact_events:
                if high_impact_event.lower() in event_lower:
                    return True
            
            # Currency-specific checks
            if 'USD' in pair:
                usd_events = ['fed', 'fomc', 'non farm', 'nfp', 'unemployment', 'gdp', 'cpi', 'ppi', 
                             'retail sales', 'home sales', 'trade balance', 'consumer confidence', 'ism']
                for event in usd_events:
                    if event in event_lower:
                        return True
            
            if 'EUR' in pair:
                eur_events = ['ecb', 'european', 'eurozone', 'german', 'france', 'italy', 'spain']
                for event in eur_events:
                    if event in event_lower:
                        return True
            
            if 'GBP' in pair:
                gbp_events = ['boe', 'bank of england', 'uk ', 'british', 'pound']
                for event in gbp_events:
                    if event in event_lower:
                        return True
            
            if 'JPY' in pair:
                jpy_events = ['boj', 'bank of japan', 'japan', 'japanese', 'yen']
                for event in jpy_events:
                    if event in event_lower:
                        return True
            
            if 'AUD' in pair:
                aud_events = ['rba', 'reserve bank of australia', 'australia', 'australian']
                for event in aud_events:
                    if event in event_lower:
                        return True
            
            if 'CAD' in pair:
                cad_events = ['boc', 'bank of canada', 'canada', 'canadian']
                for event in cad_events:
                    if event in event_lower:
                        return True
            
            if 'NZD' in pair:
                nzd_events = ['rbnz', 'reserve bank of new zealand', 'new zealand']
                for event in nzd_events:
                    if event in event_lower:
                        return True
            
            if 'CHF' in pair:
                chf_events = ['snb', 'swiss national bank', 'switzerland', 'swiss']
                for event in chf_events:
                    if event in event_lower:
                        return True
            
            return False
            
        except Exception as e:
            print(f"Error checking event affects pair: {e}")
            return False
    
    def _check_correlated_positions_limit(self, positions: List[Dict]) -> bool:
        """Check if correlated positions are within limits"""
        try:
            # Group positions by correlation groups
            correlation_groups = {
                'forex': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'USDCHF'],
                'metals': ['XAUUSD', 'XAGUSD'],
                'indices': ['US30', 'US100', 'US500', 'USTEC'],
                'crypto': ['BTCUSD', 'ETHUSD'],
                'commodities': ['USOIL']
            }
            
            group_counts = {group: 0 for group in correlation_groups}
            
            for pos in positions:
                pair = pos.get('pair', '')
                for group, pairs in correlation_groups.items():
                    if pair in pairs:
                        group_counts[group] += 1
                        break
            
            # Check if any group exceeds the limit
            return all(count <= self.risk_limits['max_correlated_positions'] 
                      for count in group_counts.values())
            
        except Exception as e:
            print(f"Error checking correlated positions limit: {e}")
            return False
    
    def _check_leverage_limits(self, positions: List[Dict], account_balance: float) -> bool:
        """Check if leverage is within limits"""
        try:
            total_position_value = sum(
                pos.get('position_size', 0) * pos.get('entry_price', 0) 
                for pos in positions
            )
            
            current_leverage = total_position_value / account_balance if account_balance > 0 else 0
            
            return current_leverage <= self.risk_limits['max_leverage']
            
        except Exception as e:
            print(f"Error checking leverage limits: {e}")
            return False
    
    def record_position(self, pair: str, direction: str, entry_price: float, 
                       position_size: float, stop_loss: float, take_profit: float,
                       risk_amount: float, risk_percentage: float, 
                       account_balance: float, market_regime: str, 
                       signal_confidence: float, metadata: Dict = None):
        """Record a new position in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO positions 
                (pair, direction, entry_price, position_size, stop_loss, take_profit,
                 risk_amount, risk_percentage, account_balance, market_regime, 
                 signal_confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pair, direction, entry_price, position_size, stop_loss, take_profit,
                risk_amount, risk_percentage, account_balance, market_regime,
                signal_confidence, json.dumps(metadata) if metadata else None
            ))
            
            position_id = cursor.lastrowid
            
            # Update daily risk
            self._update_daily_risk(risk_amount)
            
            conn.commit()
            conn.close()
            
            return position_id
            
        except Exception as e:
            print(f"Error recording position: {e}")
            return None
    
    def _update_daily_risk(self, risk_amount: float):
        """Update daily risk exposure"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = datetime.now().date()
            
            # Check if record exists
            cursor.execute('SELECT id FROM daily_risk WHERE date = ?', (today,))
            result = cursor.fetchone()
            
            if result:
                # Update existing record
                cursor.execute('''
                    UPDATE daily_risk 
                    SET total_risk = total_risk + ?
                    WHERE date = ?
                ''', (risk_amount, today))
            else:
                # Create new record
                cursor.execute('''
                    INSERT INTO daily_risk (date, total_risk)
                    VALUES (?, ?)
                ''', (today, risk_amount))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error updating daily risk: {e}")
    
    def update_position_outcome(self, position_id: int, exit_price: float, 
                              pnl: float, outcome: str, metadata: Dict = None):
        """Update position outcome in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE positions 
                SET exit_price = ?, pnl = ?, outcome = ?, exit_timestamp = ?, metadata = ?
                WHERE id = ?
            ''', (exit_price, pnl, outcome, datetime.now(), 
                  json.dumps(metadata) if metadata else None, position_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error updating position outcome: {e}")
    
    def get_risk_metrics(self, days: int = 30) -> Dict[str, float]:
        """Get comprehensive risk metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get positions from the last N days
            start_date = datetime.now() - timedelta(days=days)
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_positions,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as winning_positions,
                    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losing_positions,
                    AVG(risk_percentage) as avg_risk_percentage,
                    AVG(pnl) as avg_pnl,
                    SUM(pnl) as total_pnl,
                    MAX(risk_amount) as max_risk_amount,
                    AVG(signal_confidence) as avg_confidence
                FROM positions 
                WHERE timestamp >= ? AND status = 'closed'
            ''', (start_date,))
            
            result = cursor.fetchone()
            
            # Get daily risk data
            cursor.execute('''
                SELECT 
                    AVG(total_risk) as avg_daily_risk,
                    MAX(total_risk) as max_daily_risk,
                    SUM(total_pnl) as total_period_pnl
                FROM daily_risk 
                WHERE date >= ?
            ''', (start_date.date(),))
            
            daily_result = cursor.fetchone()
            
            conn.close()
            
            if not result or not result[0]:
                return {
                    'total_positions': 0,
                    'win_rate': 0.0,
                    'avg_risk_percentage': 0.0,
                    'avg_pnl': 0.0,
                    'total_pnl': 0.0,
                    'max_risk_amount': 0.0,
                    'avg_confidence': 0.0,
                    'avg_daily_risk': 0.0,
                    'max_daily_risk': 0.0,
                    'profit_factor': 0.0,
                    'sharpe_ratio': 0.0
                }
            
            total_positions = result[0] or 0
            winning_positions = result[1] or 0
            losing_positions = result[2] or 0
            
            win_rate = winning_positions / total_positions if total_positions > 0 else 0.0
            avg_risk_percentage = result[3] or 0.0
            avg_pnl = result[4] or 0.0
            total_pnl = result[5] or 0.0
            max_risk_amount = result[6] or 0.0
            avg_confidence = result[7] or 0.0
            
            avg_daily_risk = daily_result[0] or 0.0
            max_daily_risk = daily_result[1] or 0.0
            total_period_pnl = daily_result[2] or 0.0
            
            # Calculate profit factor
            gross_profit = sum(pos.get('pnl', 0) for pos in self._get_positions_data(days) if pos.get('pnl', 0) > 0)
            gross_loss = abs(sum(pos.get('pnl', 0) for pos in self._get_positions_data(days) if pos.get('pnl', 0) < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate Sharpe ratio (simplified)
            pnl_std = self._calculate_pnl_std(days)
            sharpe_ratio = (avg_pnl / pnl_std) if pnl_std > 0 else 0.0
            
            return {
                'total_positions': total_positions,
                'win_rate': win_rate,
                'avg_risk_percentage': avg_risk_percentage,
                'avg_pnl': avg_pnl,
                'total_pnl': total_pnl,
                'max_risk_amount': max_risk_amount,
                'avg_confidence': avg_confidence,
                'avg_daily_risk': avg_daily_risk,
                'max_daily_risk': max_daily_risk,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio
            }
            
        except Exception as e:
            print(f"Error getting risk metrics: {e}")
            return {
                'total_positions': 0,
                'win_rate': 0.0,
                'avg_risk_percentage': 0.0,
                'avg_pnl': 0.0,
                'total_pnl': 0.0,
                'max_risk_amount': 0.0,
                'avg_confidence': 0.0,
                'avg_daily_risk': 0.0,
                'max_daily_risk': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0
            }
    
    def _get_positions_data(self, days: int) -> List[Dict]:
        """Get positions data for calculations"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_date = datetime.now() - timedelta(days=days)
            
            cursor.execute('''
                SELECT pnl FROM positions 
                WHERE timestamp >= ? AND status = 'closed' AND pnl IS NOT NULL
            ''', (start_date,))
            
            results = cursor.fetchall()
            conn.close()
            
            return [{'pnl': result[0]} for result in results]
            
        except Exception as e:
            print(f"Error getting positions data: {e}")
            return []
    
    def _calculate_pnl_std(self, days: int) -> float:
        """Calculate standard deviation of PnL"""
        try:
            positions = self._get_positions_data(days)
            pnls = [pos['pnl'] for pos in positions if pos['pnl'] is not None]
            
            if len(pnls) < 2:
                return 1.0  # Default standard deviation
            
            mean_pnl = sum(pnls) / len(pnls)
            variance = sum((pnl - mean_pnl) ** 2 for pnl in pnls) / len(pnls)
            
            return math.sqrt(variance)
            
        except Exception as e:
            print(f"Error calculating PnL standard deviation: {e}")
            return 1.0
    
    def should_stop_trading(self, account_balance: float, current_drawdown: float) -> Tuple[bool, str]:
        """Determine if trading should be stopped due to risk limits"""
        try:
            # Check maximum drawdown limit (10%)
            if current_drawdown >= 0.10:
                return True, "Maximum drawdown limit reached (10%)"
            
            # Check daily loss limit (5%)
            daily_risk = self._get_daily_risk_exposure()
            if daily_risk >= self.risk_limits['max_daily_risk']:
                return True, "Daily risk limit reached"
            
            # Check account balance minimum
            if account_balance < self.risk_limits['min_account_balance']:
                return True, "Account balance below minimum"
            
            # Check consecutive losses (simplified - would need more sophisticated tracking)
            recent_positions = self._get_recent_positions(10)
            if len(recent_positions) >= 5:
                recent_outcomes = [pos.get('outcome') for pos in recent_positions[-5:]]
                if all(outcome == 'loss' for outcome in recent_outcomes):
                    return True, "5 consecutive losses - pause trading"
            
            return False, "Trading allowed"
            
        except Exception as e:
            print(f"Error checking if trading should stop: {e}")
            return True, "Error in risk assessment"
    
    def should_enter_trade(self, expected_value: float, signal_confidence: float, 
                          market_regime: str, pair: str) -> Tuple[bool, str]:
        """EV-gating: Determine if trade should be entered based on expected value"""
        try:
            # EV-gating: Only enter trades with positive expected value
            if expected_value <= 0:
                return False, f"Trade rejected: Negative expected value ({expected_value:.2f})"
            
            # Minimum confidence threshold
            if signal_confidence < 0.4:
                return False, f"Trade rejected: Low signal confidence ({signal_confidence:.2f})"
            
            # Market regime restrictions
            if market_regime == MarketRegime.HIGH_VOLATILITY.value:
                # Require higher confidence in high volatility
                if signal_confidence < 0.6:
                    return False, f"Trade rejected: Low confidence in high volatility regime"
            
            # Pair-specific restrictions
            if pair in ['BTCUSD', 'ETHUSD'] and signal_confidence < 0.5:
                return False, f"Trade rejected: Low confidence for crypto pair {pair}"
            
            # Economic event restrictions
            if self._has_upcoming_high_impact_event(pair):
                if signal_confidence < 0.7:
                    return False, f"Trade rejected: High-impact event approaching for {pair}"
            
            return True, f"Trade approved: EV={expected_value:.2f}, Confidence={signal_confidence:.2f}"
            
        except Exception as e:
            print(f"Error in EV-gating: {e}")
            return False, "Trade rejected: Error in risk assessment"
    
    def _has_upcoming_high_impact_event(self, pair: str) -> bool:
        """Check if there are upcoming high-impact economic events"""
        try:
            # This would integrate with economic calendar data
            # For now, implement basic logic based on common event times
            current_time = datetime.now()
            
            # Check for common high-impact event times (simplified)
            high_impact_hours = [8, 14, 16]  # Typical high-impact news hours
            
            # Check if we're within 1 hour of a high-impact event
            for hour in high_impact_hours:
                event_time = current_time.replace(hour=hour, minute=0, second=0, microsecond=0)
                time_diff = abs((current_time - event_time).total_seconds() / 3600)
                if time_diff <= 1:
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking for upcoming events: {e}")
            return False
    
    def _get_recent_positions(self, limit: int) -> List[Dict]:
        """Get recent positions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT outcome FROM positions 
                WHERE status = 'closed' 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            return [{'outcome': result[0]} for result in results]
            
        except Exception as e:
            print(f"Error getting recent positions: {e}")
            return []