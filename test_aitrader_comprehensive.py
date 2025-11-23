#!/usr/bin/env python3
"""Comprehensive test for aitrader functionality"""

import sys
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    print("🧪 Testing aitrader functionality step by step...")
    
    # Test 1: Import all required modules
    print("\n1. Testing imports...")
    from advanced_trading_ai import AdvancedTradingAI
    from chart_generator import ChartGenerator
    from forex_indicators import SimpleIndicators
    print("✅ All imports successful")
    
    # Test 2: Initialize components
    print("\n2. Testing component initialization...")
    ai = AdvancedTradingAI()
    chart_gen = ChartGenerator()
    print("✅ Components initialized successfully")
    
    # Test 3: Create realistic market data
    print("\n3. Creating realistic market data...")
    import time
    import random
    
    # Create more realistic price data
    base_price = 1.0850
    quotes = []
    current_time = int(time.time())
    
    # Generate realistic price movements
    price = base_price
    for i in range(200):
        # Add some realistic price volatility
        change = (random.random() - 0.5) * 0.002  # ±0.1% max change
        price += change
        
        # Create OHLC data
        open_price = price
        high_price = open_price + abs(random.gauss(0, 0.0003))
        low_price = open_price - abs(random.gauss(0, 0.0003))
        close_price = low_price + random.random() * (high_price - low_price)
        
        quote = {
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': random.randint(1000, 5000),
            'timestamp': current_time - (200 - i) * 3600,
            'time': datetime.fromtimestamp(current_time - (200 - i) * 3600).strftime('%Y-%m-%d %H:%M:%S')
        }
        quotes.append(quote)
    
    print(f"✅ Created {len(quotes)} realistic price quotes")
    print(f"   Price range: {min(q['low'] for q in quotes):.5f} - {max(q['high'] for q in quotes):.5f}")
    
    # Test 4: ML Prediction
    print("\n4. Testing ML prediction...")
    prediction = ai.predict_with_confidence(quotes)
    print(f"✅ ML prediction successful:")
    print(f"   Signal: {prediction['signal']} ({'LONG' if prediction['signal'] > 0 else 'SHORT'})")
    print(f"   Confidence: {prediction['confidence']*100:.1f}%")
    print(f"   Probability: {prediction['probability']*100:.1f}%")
    
    # Test 5: Chart generation
    print("\n5. Testing chart generation...")
    chart_bytes = chart_gen.create_technical_chart(quotes, 'EURUSD')
    if chart_bytes:
        print(f"✅ Chart generated successfully")
        print(f"   Size: {len(chart_bytes)} bytes")
    else:
        print("❌ Chart generation failed")
    
    # Test 6: Technical analysis
    print("\n6. Testing technical indicators...")
    closes = [q['close'] for q in quotes]
    
    sma_10 = SimpleIndicators.sma(closes, 10)
    sma_20 = SimpleIndicators.sma(closes, 20)
    rsi_14 = SimpleIndicators.rsi(closes, 14)
    macd_data = SimpleIndicators.macd(closes)
    bb_data = SimpleIndicators.bollinger_bands(closes, 20)
    atr_values = SimpleIndicators.atr(quotes, 14)
    
    print(f"✅ Technical indicators calculated:")
    print(f"   SMA(10): {sma_10[-1]:.5f}")
    print(f"   SMA(20): {sma_20[-1]:.5f}")
    print(f"   RSI(14): {rsi_14[-1]:.1f}")
    print(f"   MACD: {macd_data['macd'][-1]:.5f}")
    print(f"   ATR(14): {atr_values[-1]:.5f}")
    print(f"   BB Upper: {bb_data['upper'][-1]:.5f}")
    print(f"   BB Lower: {bb_data['lower'][-1]:.5f}")
    
    # Test 7: Model stats
    print("\n7. Testing model statistics...")
    stats = ai.get_model_stats()
    print(f"✅ Model stats retrieved:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n🎉 All tests completed successfully!")
    print("\n💡 Recommendations for improvement:")
    print("   - Consider adding more sophisticated ML training")
    print("   - Add more technical indicators for better accuracy")
    print("   - Implement real-time data validation")
    print("   - Add error handling for edge cases")
    
except Exception as e:
    print(f"\n❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)