#!/usr/bin/env python3
"""Test script for aitrader functionality"""

import sys
import os

try:
    # Test imports
    print("Testing imports...")
    from advanced_trading_ai import AdvancedTradingAI
    from chart_generator import AdvancedChartGenerator
    from forex_indicators import AdvancedTechnicalIndicators
    print("✅ All imports successful")
    
    # Test AdvancedTradingAI initialization
    print("Testing AdvancedTradingAI initialization...")
    ai = AdvancedTradingAI()
    print("✅ AdvancedTradingAI initialized successfully")
    
    # Test ChartGenerator initialization
    print("Testing ChartGenerator initialization...")
    chart_gen = AdvancedChartGenerator()
    print("✅ ChartGenerator initialized successfully")
    
    # Test with sample data
    print("Testing with sample data...")
    import time
    sample_quotes = []
    base_price = 1.0800
    current_time = int(time.time())
    for i in range(100):
        quote = {
            'open': base_price + (i * 0.0001) + (i % 10) * 0.0002,
            'high': base_price + (i * 0.0001) + (i % 10) * 0.0003,
            'low': base_price + (i * 0.0001) - (i % 10) * 0.0001,
            'close': base_price + (i * 0.0001) + (i % 8) * 0.0002,
            'volume': 1000 + i * 10,
            'time': f'2024-01-01 {i:02d}:00:00',
            'timestamp': current_time - (100 - i) * 3600  # часовые данные
        }
        sample_quotes.append(quote)
    
    # Test ML prediction
    print("Testing ML prediction...")
    prediction = ai.predict_with_confidence(sample_quotes)
    print(f"✅ ML prediction: {prediction}")
    
    # Test chart generation
    print("Testing chart generation...")
    chart_bytes = chart_gen.create_technical_chart(sample_quotes, 'EURUSD')
    if chart_bytes:
        print(f"✅ Chart generated successfully, size: {len(chart_bytes)} bytes")
    else:
        print("❌ Chart generation failed")
    
    print("\n🎉 All tests passed! Aitrader functionality is working.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)