import asyncio
import json
import sqlite3
import requests
import yfinance as yf
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Any
from binance import AsyncClient, BinanceSocketManager
from binance.enums import FuturesType
import websocket
import threading
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os

@dataclass
class MarketData:
    """Market data structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    pair: str
    timeframe: str

class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    async def get_historical_data(self, pair: str, timeframe: str, limit: int = 100) -> List[Dict]:
        """Get historical market data"""
        pass
    
    @abstractmethod
    async def start_realtime_stream(self, pair: str, timeframe: str, callback: Callable):
        """Start real-time data stream"""
        pass
    
    @abstractmethod
    async def stop_realtime_stream(self):
        """Stop real-time data stream"""
        pass

class BinanceDataSource(DataSource):
    """Binance data source for crypto trading"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = None
        self.socket_manager = None
        self.streams = {}
        self.running = False
        
        # Timeframe mappings
        self.timeframe_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
        }
    
    async def initialize(self):
        """Initialize Binance client"""
        try:
            if self.testnet:
                self.client = await AsyncClient.create(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    testnet=True
                )
            else:
                self.client = await AsyncClient.create(
                    api_key=self.api_key,
                    api_secret=self.api_secret
                )
            
            self.socket_manager = BinanceSocketManager(self.client)
            return True
        except Exception as e:
            print(f"Error initializing Binance client: {e}")
            return False
    
    async def get_historical_data(self, pair: str, timeframe: str, limit: int = 100) -> List[Dict]:
        """Get historical kline data from Binance"""
        try:
            if not self.client:
                await self.initialize()
            
            # Normalize pair format
            symbol = pair.replace('/', '').upper()
            interval = self.timeframe_map.get(timeframe, '1h')
            
            # Get klines
            klines = await self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # Convert to standard format
            quotes = []
            for kline in klines:
                quotes.append({
                    'timestamp': datetime.fromtimestamp(kline[0] / 1000).isoformat(),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'pair': pair,
                    'timeframe': timeframe
                })
            
            return quotes
            
        except Exception as e:
            print(f"Error getting historical data from Binance: {e}")
            return []
    
    async def start_realtime_stream(self, pair: str, timeframe: str, callback: Callable):
        """Start real-time kline stream"""
        try:
            if not self.socket_manager:
                await self.initialize()
            
            symbol = pair.replace('/', '').upper()
            interval = self.timeframe_map.get(timeframe, '1h')
            stream_key = f"{symbol}@{interval}"
            
            if stream_key in self.streams:
                print(f"Stream {stream_key} already running")
                return
            
            # Create kline socket
            kline_socket = self.socket_manager.kline_socket(symbol=symbol, interval=interval)
            
            async def process_kline(msg):
                """Process incoming kline data"""
                try:
                    kline = msg['k']
                    
                    # Create market data
                    market_data = MarketData(
                        timestamp=datetime.fromtimestamp(kline['t'] / 1000),
                        open=float(kline['o']),
                        high=float(kline['h']),
                        low=float(kline['l']),
                        close=float(kline['c']),
                        volume=float(kline['v']),
                        pair=pair,
                        timeframe=timeframe
                    )
                    
                    # Convert to dict format for callback
                    quote = {
                        'timestamp': market_data.timestamp.isoformat(),
                        'open': market_data.open,
                        'high': market_data.high,
                        'low': market_data.low,
                        'close': market_data.close,
                        'volume': market_data.volume,
                        'pair': pair,
                        'timeframe': timeframe,
                        'is_final': kline['x']
                    }
                    
                    # Call callback
                    await callback(quote)
                    
                except Exception as e:
                    print(f"Error processing kline data: {e}")
            
            # Start stream
            self.streams[stream_key] = kline_socket
            
            async with kline_socket as stream:
                self.running = True
                while self.running:
                    msg = await stream.recv()
                    await process_kline(msg)
                    
        except Exception as e:
            print(f"Error starting Binance stream: {e}")
    
    async def stop_realtime_stream(self):
        """Stop all real-time streams"""
        self.running = False
        self.streams.clear()
        
        if self.client:
            await self.client.close_connection()
        
        print("Binance streams stopped")

class YahooFinanceDataSource(DataSource):
    """Yahoo Finance data source for forex, stocks, and commodities"""
    
    def __init__(self):
        self.running = False
        self.streams = {}
        
        # Pair mappings for Yahoo Finance
        self.pair_map = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X',
            'USDCHF': 'USDCHF=X',
            'NZDUSD': 'NZDUSD=X',
            'XAUUSD': 'GC=F',  # Gold futures
            'XAGUSD': 'SI=F',  # Silver futures
            'USOIL': 'CL=F',   # Crude oil futures
            'BTCUSD': 'BTC-USD',
            'ETHUSD': 'ETH-USD',
            'US30': '^DJI',
            'US100': '^IXIC',
            'US500': '^GSPC',
            'USTEC': '^IXIC',  # Same as US100
            'AAPL': 'AAPL',
            'TSLA': 'TSLA'
        }
    
    async def get_historical_data(self, pair: str, timeframe: str, limit: int = 100) -> List[Dict]:
        """Get historical data from Yahoo Finance"""
        try:
            # Map pair to Yahoo Finance symbol
            symbol = self.pair_map.get(pair, pair)
            
            days = 5
            if timeframe in ['1m','5m','15m','30m','1h','4h']:
                if timeframe == '1m': days = min(7, max(1, int(limit/60/24))) or 1
                elif timeframe == '5m': days = min(30, max(1, int(limit/12/24))) or 7
                elif timeframe == '15m': days = min(60, max(1, int(limit/4/24))) or 7
                elif timeframe == '30m': days = min(60, max(1, int(limit/2/24))) or 7
                elif timeframe == '1h': days = min(30, max(1, int(limit/24))) or 7
                elif timeframe == '4h': days = min(60, max(1, int(limit/6))) or 30
            else:
                days = min(3650, max(1, int(limit)))
            period = f'{days}d'
            
            # Get data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=timeframe)
            
            # Convert to standard format
            quotes = []
            for index, row in hist.iterrows():
                quotes.append({
                    'timestamp': index.isoformat(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume']) if 'Volume' in row else 0.0,
                    'pair': pair,
                    'timeframe': timeframe
                })
            
            return quotes
            
        except Exception as e:
            print(f"Error getting historical data from Yahoo Finance: {e}")
            return []
    
    async def start_realtime_stream(self, pair: str, timeframe: str, callback: Callable):
        """Start real-time data stream (simulated for Yahoo Finance)"""
        try:
            symbol = self.pair_map.get(pair, pair)
            stream_key = f"{symbol}_{timeframe}"
            
            if stream_key in self.streams:
                print(f"Stream {stream_key} already running")
                return
            
            async def stream_data():
                """Simulated real-time stream using periodic data fetching"""
                self.running = True
                last_price = None
                
                while self.running:
                    try:
                        # Get latest data
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        
                        # Get current price
                        current_price = info.get('currentPrice', info.get('regularMarketPrice', last_price))
                        
                        if current_price and current_price != last_price:
                            # Create synthetic quote
                            quote = {
                                'timestamp': datetime.now().isoformat(),
                                'open': current_price,
                                'high': current_price * 1.001,
                                'low': current_price * 0.999,
                                'close': current_price,
                                'volume': 1000,  # Simulated volume
                                'pair': pair,
                                'timeframe': timeframe,
                                'is_final': True
                            }
                            
                            await callback(quote)
                            last_price = current_price
                        
                        # Wait based on timeframe
                        sleep_time = {
                            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
                            '1h': 3600, '4h': 14400, '1d': 86400
                        }.get(timeframe, 60)
                        
                        await asyncio.sleep(sleep_time)
                        
                    except Exception as e:
                        print(f"Error in Yahoo Finance stream: {e}")
                        await asyncio.sleep(60)  # Wait 1 minute on error
            
            # Start stream task
            self.streams[stream_key] = asyncio.create_task(stream_data())
            
        except Exception as e:
            print(f"Error starting Yahoo Finance stream: {e}")
    
    async def stop_realtime_stream(self):
        """Stop all real-time streams"""
        self.running = False
        
        # Cancel all stream tasks
        for task in self.streams.values():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.streams.clear()
        print("Yahoo Finance streams stopped")

class StooqDataSource(DataSource):
    def __init__(self):
        self.base_url = "https://stooq.com/q/d/l/"
        self.symbol_map = {
            'EURUSD': 'eurusd',
            'GBPUSD': 'gbpusd',
            'XAUUSD': 'xauusd',
            'USOIL': 'wtic',
            'AAPL': 'aapl.us',
        }
    async def get_historical_data(self, pair: str, timeframe: str, limit: int = 100) -> List[Dict]:
        try:
            s = self.symbol_map.get(pair, pair.lower())
            i = 'd'
            if timeframe in ['1h','4h']:
                i = 'd'
            params = { 's': s, 'i': i }
            r = requests.get(self.base_url, params=params, timeout=10)
            r.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            if 'Date' not in df.columns:
                return []
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
            df = df.dropna()
            if limit and len(df) > limit:
                df = df.tail(limit)
            quotes = []
            for _, row in df.iterrows():
                quotes.append({
                    'timestamp': row['Date'].isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']) if 'volume' in df.columns else 0.0,
                    'pair': pair,
                    'timeframe': timeframe
                })
            return quotes
        except Exception as e:
            print(f"Error getting historical data from Stooq: {e}")
            return []
    async def start_realtime_stream(self, pair: str, timeframe: str, callback: Callable):
        return
    async def stop_realtime_stream(self):
        return

class AlphaVantageDataSource(DataSource):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
    async def get_historical_data(self, pair: str, timeframe: str, limit: int = 100) -> List[Dict]:
        try:
            if len(pair) <= 5 and pair.isalpha():
                func = 'TIME_SERIES_INTRADAY'
                interval = '60min' if timeframe in ['1h','4h'] else '15min'
                url = 'https://www.alphavantage.co/query'
                params = { 'function': func, 'symbol': pair, 'interval': interval, 'apikey': self.api_key, 'outputsize': 'compact' }
                resp = self.session.get(url, params=params, timeout=10)
                data = resp.json()
                key = f"Time Series ({interval})"
                series = data.get(key, {})
                rows = list(series.items())
                rows.sort(key=lambda x: x[0])
                if limit:
                    rows = rows[-limit:]
                quotes = []
                for ts, row in rows:
                    quotes.append({
                        'timestamp': datetime.fromisoformat(ts).isoformat(),
                        'open': float(row['1. open']),
                        'high': float(row['2. high']),
                        'low': float(row['3. low']),
                        'close': float(row['4. close']),
                        'volume': float(row.get('5. volume', 0)),
                        'pair': pair,
                        'timeframe': timeframe
                    })
                return quotes
            else:
                func = 'FX_INTRADAY'
                interval = '60min' if timeframe in ['1h','4h'] else '15min'
                from_symbol = pair[:3]
                to_symbol = pair[3:]
                url = 'https://www.alphavantage.co/query'
                params = { 'function': func, 'from_symbol': from_symbol, 'to_symbol': to_symbol, 'interval': interval, 'apikey': self.api_key } 
                resp = self.session.get(url, params=params, timeout=10)
                data = resp.json()
                key = f"Time Series FX ({interval})"
                series = data.get(key, {})
                rows = list(series.items())
                rows.sort(key=lambda x: x[0])
                if limit:
                    rows = rows[-limit:]
                quotes = []
                for ts, row in rows:
                    quotes.append({
                        'timestamp': datetime.fromisoformat(ts).isoformat(),
                        'open': float(row['1. open']),
                        'high': float(row['2. high']),
                        'low': float(row['3. low']),
                        'close': float(row['4. close']),
                        'volume': 0.0,
                        'pair': pair,
                        'timeframe': timeframe
                    })
                return quotes
        except Exception as e:
            print(f"Error getting AlphaVantage data: {e}")
            return []
    async def start_realtime_stream(self, pair: str, timeframe: str, callback: Callable):
        return
    async def stop_realtime_stream(self):
        return

class HistDataLocalSource(DataSource):
    def __init__(self, base_path: str = 'data/histdata'):
        self.base_path = base_path
    async def get_historical_data(self, pair: str, timeframe: str, limit: int = 100) -> List[Dict]:
        try:
            fname = os.path.join(self.base_path, f"{pair}_{timeframe}.csv")
            if not os.path.exists(fname):
                return []
            df = pd.read_csv(fname)
            cols = { 'Timestamp':'timestamp','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume' }
            df = df.rename(columns=cols)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.dropna()
            if limit and len(df) > limit:
                df = df.tail(limit)
            quotes = []
            for _, row in df.iterrows():
                ts = row['timestamp'] if isinstance(row['timestamp'], datetime) else pd.to_datetime(row['timestamp'])
                quotes.append({
                    'timestamp': ts.isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row.get('volume', 0.0)),
                    'pair': pair,
                    'timeframe': timeframe
                })
            return quotes
        except Exception as e:
            print(f"Error reading HistData local file: {e}")
            return []
    async def start_realtime_stream(self, pair: str, timeframe: str, callback: Callable):
        return
    async def stop_realtime_stream(self):
        return

class TrueFXDataSource(DataSource):
    def __init__(self, base_path: str = 'data/truefx'):
        self.base_path = base_path
    async def get_historical_data(self, pair: str, timeframe: str, limit: int = 100) -> List[Dict]:
        try:
            fname = os.path.join(self.base_path, f"{pair.replace('/','')}_{timeframe}.csv")
            if not os.path.exists(fname):
                return []
            df = pd.read_csv(fname)
            cols = { 'Timestamp':'timestamp','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume' }
            df = df.rename(columns=cols)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.dropna()
            if limit and len(df) > limit:
                df = df.tail(limit)
            quotes = []
            for _, row in df.iterrows():
                ts = row['timestamp'] if isinstance(row['timestamp'], datetime) else pd.to_datetime(row['timestamp'])
                quotes.append({
                    'timestamp': ts.isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row.get('volume', 0.0)),
                    'pair': pair,
                    'timeframe': timeframe
                })
            return quotes
        except Exception as e:
            print(f"Error reading TrueFX local file: {e}")
            return []
    async def start_realtime_stream(self, pair: str, timeframe: str, callback: Callable):
        return
    async def stop_realtime_stream(self):
        return
class KaikoDataSource(DataSource):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
    async def get_historical_data(self, pair: str, timeframe: str, limit: int = 100) -> List[Dict]:
        try:
            return []
        except Exception as e:
            print(f"Error getting Kaiko data: {e}")
            return []
    async def start_realtime_stream(self, pair: str, timeframe: str, callback: Callable):
        return
    async def stop_realtime_stream(self):
        return

class TradingEconomicsDataSource:
    """Trading Economics data source for economic calendar and indicators"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tradingeconomics.com"
        self.session = requests.Session()
        
        # Cache for economic events
        self.events_cache = {}
        self.cache_timeout = 3600  # 1 hour
    
    async def get_economic_calendar(self, country: str = 'all', 
                                  importance: str = 'all',
                                  start_date: datetime = None,
                                  end_date: datetime = None) -> List[Dict]:
        """Get economic calendar events"""
        try:
            # Default date range: next 7 days
            if not start_date:
                start_date = datetime.now()
            if not end_date:
                end_date = start_date + timedelta(days=7)
            
            # Build URL
            url = f"{self.base_url}/calendar"
            params = {
                'c': self.api_key,
                'country': country,
                'importance': importance,
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            }
            
            # Make request
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            events = response.json()
            
            # Filter and format events
            formatted_events = []
            for event in events:
                formatted_events.append({
                    'id': event.get('CalendarId'),
                    'country': event.get('Country'),
                    'category': event.get('Category'),
                    'event': event.get('Event'),
                    'reference': event.get('Reference'),
                    'source': event.get('Source'),
                    'actual': event.get('Actual'),
                    'previous': event.get('Previous'),
                    'forecast': event.get('Forecast'),
                    'importance': event.get('Importance'),
                    'timestamp': event.get('Date'),
                    'impact': self._calculate_impact(event)
                })
            
            return formatted_events
            
        except Exception as e:
            print(f"Error getting economic calendar: {e}")
            return []
    
    def _calculate_impact(self, event: Dict) -> str:
        """Calculate event impact level"""
        importance = event.get('Importance', '').lower()
        category = event.get('Category', '').lower()
        
        # High impact events
        high_impact_categories = [
            'interest rate', 'inflation', 'gdp', 'employment', 'unemployment',
            'non farm', 'fed', 'ecb', 'boe', 'boj', 'snb'
        ]
        
        if importance == 'high' or any(cat in category for cat in high_impact_categories):
            return 'high'
        elif importance == 'medium':
            return 'medium'
        elif importance == 'low':
            return 'low'
        else:
            return 'medium'
    
    def is_high_impact_event_near(self, pair: str, minutes_before: int = 30, 
                                 minutes_after: int = 30) -> bool:
        """Check if there's a high impact event near current time"""
        try:
            # Get recent events
            events = self.get_economic_calendar(
                start_date=datetime.now() - timedelta(minutes=minutes_before),
                end_date=datetime.now() + timedelta(minutes=minutes_after)
            )
            
            # Filter high impact events
            high_impact_events = [e for e in events if e['impact'] == 'high']
            
            return len(high_impact_events) > 0
            
        except Exception as e:
            print(f"Error checking high impact events: {e}")
            return False

class FredDataSource:
    """FRED (Federal Reserve Economic Data) source for macroeconomic indicators"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        self.session = requests.Session()
        
        # Key FRED series for trading
        self.key_series = {
            'DGS10': '10-Year Treasury Constant Maturity Rate',
            'DXY': 'US Dollar Index',  # Note: This might need different series
            'VIXCLS': 'CBOE Volatility Index',
            'CPIAUCSL': 'Consumer Price Index',
            'UNRATE': 'Unemployment Rate',
            'GDPC1': 'Real Gross Domestic Product',
            'FEDFUNDS': 'Federal Funds Rate'
        }
    
    async def get_series_data(self, series_id: str, 
                            start_date: datetime = None,
                            end_date: datetime = None) -> List[Dict]:
        """Get FRED series data"""
        try:
            # Default date range: last 30 days
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            # Build URL
            url = f"{self.base_url}/series/observations"
            params = {
                'api_key': self.api_key,
                'series_id': series_id,
                'file_type': 'json',
                'observation_start': start_date.strftime('%Y-%m-%d'),
                'observation_end': end_date.strftime('%Y-%m-%d')
            }
            
            # Make request
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Format observations
            observations = []
            for obs in data.get('observations', []):
                if obs.get('value') and obs['value'] != '.':
                    observations.append({
                        'date': obs['date'],
                        'value': float(obs['value']),
                        'series_id': series_id
                    })
            
            return observations
            
        except Exception as e:
            print(f"Error getting FRED series data: {e}")
            return []

class DataManager:
    """Central data manager that coordinates multiple data sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sources = {}
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        
        # Initialize data sources
        self._initialize_sources()
        
        # Database for caching
        self.db_path = config.get('database_path', 'market_data.db')
        self.init_database()
    
    def _initialize_sources(self):
        """Initialize all data sources"""
        # Binance for crypto
        if self.config.get('binance_api_key'):
            self.sources['binance'] = BinanceDataSource(
                api_key=self.config['binance_api_key'],
                api_secret=self.config.get('binance_api_secret'),
                testnet=self.config.get('binance_testnet', True)
            )
        
        # Yahoo Finance for forex/stocks/commodities
        self.sources['yahoo'] = YahooFinanceDataSource()
        
        # Stooq
        self.sources['stooq'] = StooqDataSource()
        
        # AlphaVantage
        if self.config.get('alphavantage_api_key'):
            self.sources['alphavantage'] = AlphaVantageDataSource(self.config['alphavantage_api_key'])
        
        # HistData local
        self.sources['histdata'] = HistDataLocalSource(self.config.get('histdata_path', 'data/histdata'))

        # TrueFX local
        if self.config.get('truefx_path'):
            self.sources['truefx'] = TrueFXDataSource(self.config.get('truefx_path'))
        
        # Kaiko
        if self.config.get('kaiko_api_key'):
            self.sources['kaiko'] = KaikoDataSource(self.config['kaiko_api_key'])
        
        # Trading Economics for economic calendar
        if self.config.get('trading_economics_api_key'):
            self.sources['trading_economics'] = TradingEconomicsDataSource(
                api_key=self.config['trading_economics_api_key']
            )
        
        # FRED for macro data
        if self.config.get('fred_api_key'):
            self.sources['fred'] = FredDataSource(
                api_key=self.config['fred_api_key']
            )
    
    def init_database(self):
        """Initialize database for caching"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL,
                source TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pair, timeframe, timestamp, source)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS economic_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                country TEXT NOT NULL,
                category TEXT NOT NULL,
                event_name TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                impact TEXT NOT NULL,
                actual TEXT,
                previous TEXT,
                forecast TEXT,
                source TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(event_id, source)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def get_market_data(self, pair: str, timeframe: str, limit: int = 100, 
                            force_refresh: bool = False) -> List[Dict]:
        """Get market data from appropriate source"""
        try:
            # Check cache first
            cache_key = f"{pair}_{timeframe}_{limit}"
            if not force_refresh and cache_key in self.cache:
                cache_time, data = self.cache[cache_key]
                if datetime.now() - cache_time < timedelta(seconds=self.cache_timeout):
                    return data
            
            # Determine best source
            source = self._get_best_source(pair)
            
            if source not in self.sources:
                print(f"No suitable data source for {pair}")
                return []
            
            # Get data
            data_source = self.sources[source]
            data = await data_source.get_historical_data(pair, timeframe, limit)
            
            # Cache result
            self.cache[cache_key] = (datetime.now(), data)
            
            # Store in database
            await self._store_market_data(data, source)
            
            return data
            
        except Exception as e:
            print(f"Error getting market data: {e}")
            return []
    
    def _get_best_source(self, pair: str) -> str:
        """Determine best data source for pair"""
        pair_upper = pair.upper()
        
        # Crypto pairs
        if any(crypto in pair_upper for crypto in ['BTC', 'ETH', 'BNB', 'ADA', 'DOT']):
            if 'binance' in self.sources:
                return 'binance'
            if 'kaiko' in self.sources:
                return 'kaiko'
            return 'yahoo'
        
        # Forex pairs
        if any(currency in pair_upper for currency in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD']):
            if 'truefx' in self.sources:
                return 'truefx'
            if pair_upper in ['EURUSD','GBPUSD'] and 'histdata' in self.sources:
                return 'histdata'
            return 'yahoo' if 'yahoo' in self.sources else 'stooq'
        
        # Stock symbols
        if len(pair_upper) <= 5 and pair_upper.isalpha():
            if 'alphavantage' in self.sources:
                return 'alphavantage'
            return 'yahoo'
        
        # Default to Yahoo Finance
        return 'yahoo'

    async def get_merged_market_data(self, pair: str, timeframe: str, limit: int = 100) -> List[Dict]:
        try:
            primary = await self.get_market_data(pair, timeframe, limit)
            if not primary and 'yahoo' in self.sources:
                primary = await self.sources['yahoo'].get_historical_data(pair, timeframe, limit)
            backups = []
            for src in ['truefx','stooq','alphavantage']:
                if src in self.sources:
                    d = await self.sources[src].get_historical_data(pair, timeframe, limit)
                    if d:
                        backups.extend(d)
            by_time = {}
            for q in backups:
                by_time.setdefault(q['timestamp'], q)
            merged = []
            for q in primary:
                t = q['timestamp']
                if t in by_time and q.get('volume', 0) in [None, 0]:
                    q['volume'] = by_time[t].get('volume', 0)
                merged.append(q)
            return merged if merged else primary
        except Exception as e:
            print(f"Error merging market data: {e}")
            return []
    
    async def _store_market_data(self, data: List[Dict], source: str):
        """Store market data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for quote in data:
                cursor.execute('''
                    INSERT OR REPLACE INTO market_data 
                    (pair, timeframe, timestamp, open, high, low, close, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    quote['pair'], quote['timeframe'], quote['timestamp'],
                    quote['open'], quote['high'], quote['low'], quote['close'],
                    quote.get('volume', 0), source
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error storing market data: {e}")
    
    async def get_economic_calendar(self, country: str = 'all', 
                                  importance: str = 'all',
                                  start_date: datetime = None,
                                  end_date: datetime = None) -> List[Dict]:
        """Get economic calendar events"""
        if 'trading_economics' not in self.sources:
            return []
        
        try:
            source = self.sources['trading_economics']
            events = await source.get_economic_calendar(country, importance, start_date, end_date)
            
            # Store in database
            await self._store_economic_events(events, 'trading_economics')
            
            return events
            
        except Exception as e:
            print(f"Error getting economic calendar: {e}")
            return []
    
    async def _store_economic_events(self, events: List[Dict], source: str):
        """Store economic events in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for event in events:
                cursor.execute('''
                    INSERT OR REPLACE INTO economic_events 
                    (event_id, country, category, event_name, timestamp, impact, 
                     actual, previous, forecast, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event['id'], event['country'], event['category'], 
                    event['event'], event['timestamp'], event['impact'],
                    str(event.get('actual', '')), str(event.get('previous', '')),
                    str(event.get('forecast', '')), source
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error storing economic events: {e}")
    
    async def get_macro_indicators(self, indicators: List[str] = None) -> Dict[str, List[Dict]]:
        """Get macroeconomic indicators from FRED"""
        if 'fred' not in self.sources:
            return {}
        
        try:
            source = self.sources['fred']
            
            # Default indicators
            if not indicators:
                indicators = ['DGS10', 'VIXCLS', 'CPIAUCSL', 'UNRATE']
            
            results = {}
            for indicator in indicators:
                data = await source.get_series_data(indicator)
                results[indicator] = data
            
            return results
            
        except Exception as e:
            print(f"Error getting macro indicators: {e}")
            return {}
    
    async def start_realtime_stream(self, pair: str, timeframe: str, callback: Callable):
        """Start real-time data stream"""
        source = self._get_best_source(pair)
        
        if source not in self.sources:
            print(f"No suitable data source for {pair}")
            return
        
        data_source = self.sources[source]
        await data_source.start_realtime_stream(pair, timeframe, callback)
    
    async def stop_all_streams(self):
        """Stop all real-time streams"""
        for source in self.sources.values():
            if hasattr(source, 'stop_realtime_stream'):
                await source.stop_realtime_stream()
    
    def is_high_impact_event_near(self, pair: str, minutes_before: int = 30, 
                                 minutes_after: int = 30) -> bool:
        """Check if there's a high impact event near current time"""
        if 'trading_economics' not in self.sources:
            return False
        
        try:
            source = self.sources['trading_economics']
            return source.is_high_impact_event_near(pair, minutes_before, minutes_after)
            
        except Exception as e:
            print(f"Error checking high impact events: {e}")
            return False
    
    def get_data_sources_status(self) -> Dict[str, str]:
        """Get status of all data sources"""
        status = {}
        
        for name, source in self.sources.items():
            if hasattr(source, 'running'):
                status[name] = "running" if source.running else "stopped"
            else:
                status[name] = "available"
        
        return status
