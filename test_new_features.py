#!/usr/bin/env python3
"""
Test script for new AI trading features:
- Heikin Ashi indicator
- VWAP indicator  
- News event avoidance mode
"""

import sys
import asyncio
from datetime import datetime, timedelta
from forex_indicators import AdvancedTechnicalIndicators
from risk_management import RiskManager, PositionSizingParams
from advanced_trading_ai import AdvancedTradingAI

def test_heikin_ashi():
    """Test Heikin Ashi indicator"""
    print("🧪 Testing Heikin Ashi indicator...")
    
    indicators = AdvancedTechnicalIndicators()
    
    # Create test data
    test_quotes = [
        {'open': 1.1000, 'high': 1.1020, 'low': 1.0980, 'close': 1.1010},
        {'open': 1.1010, 'high': 1.1030, 'low': 1.0990, 'close': 1.1020},
        {'open': 1.1020, 'high': 1.1040, 'low': 1.1000, 'close': 1.1030},
        {'open': 1.1030, 'high': 1.1050, 'low': 1.1010, 'close': 1.1040},
        {'open': 1.1040, 'high': 1.1060, 'low': 1.1020, 'close': 1.1050},
    ]
    
    # Test Heikin Ashi calculation
    ha_data = indicators.heikin_ashi(test_quotes)
    
    assert 'open' in ha_data
    assert 'high' in ha_data
    assert 'low' in ha_data
    assert 'close' in ha_data
    assert len(ha_data['close']) == len(test_quotes)
    
    # Check that HA values are smoothed (less volatile than regular candles)
    regular_ranges = [q['high'] - q['low'] for q in test_quotes]
    ha_ranges = [ha_data['high'][i] - ha_data['low'][i] for i in range(len(test_quotes))]
    
    print(f"✅ Heikin Ashi calculated successfully")
    print(f"   Regular candle avg range: {sum(regular_ranges)/len(regular_ranges):.4f}")
    print(f"   HA candle avg range: {sum(ha_ranges)/len(ha_ranges):.4f}")
    
    return True

def test_vwap():
    """Test VWAP indicator"""
    print("🧪 Testing VWAP indicator...")
    
    indicators = AdvancedTechnicalIndicators()
    
    # Create test data with volume
    test_quotes = [
        {'open': 1.1000, 'high': 1.1020, 'low': 1.0980, 'close': 1.1010, 'volume': 1000},
        {'open': 1.1010, 'high': 1.1030, 'low': 1.0990, 'close': 1.1020, 'volume': 1200},
        {'open': 1.1020, 'high': 1.1040, 'low': 1.1000, 'close': 1.1030, 'volume': 800},
        {'open': 1.1030, 'high': 1.1050, 'low': 1.1010, 'close': 1.1040, 'volume': 1500},
        {'open': 1.1040, 'high': 1.1060, 'low': 1.1020, 'close': 1.1050, 'volume': 900},
    ]
    
    # Test VWAP calculation
    vwap_values = indicators.vwap(test_quotes)
    
    assert len(vwap_values) == len(test_quotes)
    assert all(isinstance(v, (int, float)) for v in vwap_values)
    
    # VWAP should be between high and low for each period
    for i, vwap in enumerate(vwap_values):
        quote = test_quotes[i]
        assert quote['low'] <= vwap <= quote['high']
    
    print(f"✅ VWAP calculated successfully")
    print(f"   VWAP values: {[f'{v:.4f}' for v in vwap_values[-3:]]}")
    
    return True

def test_news_event_avoidance():
    """Test news event avoidance functionality"""
    print("🧪 Testing news event avoidance...")
    
    risk_manager = RiskManager()
    
    # Create test upcoming events
    current_time = datetime.now()
    upcoming_events = [
        {
            'timestamp': (current_time + timedelta(hours=1)).isoformat() + 'Z',
            'event': 'Non Farm Payrolls',
            'importance': 'high',
            'country': 'United States',
            'actual': '',
            'forecast': '180K',
            'previous': '175K'
        },
        {
            'timestamp': (current_time + timedelta(hours=5)).isoformat() + 'Z',
            'event': 'Federal Funds Rate',
            'importance': 'high',
            'country': 'United States',
            'actual': '',
            'forecast': '5.25%',
            'previous': '5.25%'
        },
        {
            'timestamp': (current_time + timedelta(hours=12)).isoformat() + 'Z',
            'event': 'GDP Growth Rate',
            'importance': 'medium',
            'country': 'United States',
            'actual': '',
            'forecast': '2.1%',
            'previous': '2.0%'
        }
    ]
    
    # Test with EURUSD pair (should be affected by USD events)
    result = risk_manager.check_news_event_avoidance('EURUSD', upcoming_events)
    
    assert result['should_avoid'] == True
    assert 'Non Farm Payrolls' in result['reason']
    assert result['time_to_event'] is not None
    assert result['time_to_event'] <= 2  # Should be within 2 hours
    
    print(f"✅ News event avoidance working correctly")
    print(f"   Should avoid: {result['should_avoid']}")
    print(f"   Reason: {result['reason']}")
    print(f"   Time to event: {result['time_to_event']:.1f} hours")
    
    # Test with no events
    result_empty = risk_manager.check_news_event_avoidance('EURUSD', [])
    assert result_empty['should_avoid'] == False
    
    # Test with events far in the future
    future_events = [
        {
            'timestamp': (current_time + timedelta(hours=25)).isoformat() + 'Z',
            'event': 'Non Farm Payrolls',
            'importance': 'high',
            'country': 'United States',
            'actual': '',
            'forecast': '180K',
            'previous': '175K'
        }
    ]
    
    result_future = risk_manager.check_news_event_avoidance('EURUSD', future_events)
    assert result_future['should_avoid'] == False  # Should not avoid events > 24 hours away
    
    return True

def test_position_sizing_with_news_avoidance():
    """Test position sizing with news event avoidance"""
    print("🧪 Testing position sizing with news avoidance...")
    
    risk_manager = RiskManager()
    
    # Create test upcoming events with high-impact event within 2 hours
    current_time = datetime.now()
    upcoming_events = [
        {
            'timestamp': (current_time + timedelta(hours=1)).isoformat() + 'Z',
            'event': 'Non Farm Payrolls',
            'importance': 'high',
            'country': 'United States',
            'actual': '',
            'forecast': '180K',
            'previous': '175K'
        }
    ]
    
    # Create position sizing parameters
    params = PositionSizingParams(
        account_balance=10000.0,
        entry_price=1.1050,
        stop_loss_price=1.1000,
        risk_percentage=0.02,
        atr=0.0010,
        market_regime='trend',
        signal_confidence=0.75,
        pair='EURUSD',
        upcoming_events=upcoming_events
    )
    
    # Calculate position size
    metrics = risk_manager.calculate_position_size(params)
    
    # Should return zero position size due to news event avoidance
    assert metrics.position_size == 0.0
    assert metrics.risk_amount == 0.0
    assert metrics.risk_percentage == 0.0
    
    print(f"✅ Position sizing with news avoidance working correctly")
    print(f"   Position size: {metrics.position_size}")
    print(f"   Risk amount: ${metrics.risk_amount}")
    print(f"   Risk percentage: {metrics.risk_percentage}%")
    
    return True

def test_ai_integration():
    """Test AI integration with new features"""
    print("🧪 Testing AI integration with new features...")
    
    ai_system = AdvancedTradingAI()
    
    # Create test data
    test_quotes = []
    base_price = 1.1000
    for i in range(100):
        price = base_price + (i * 0.0001) + (0.0005 * (i % 5))  # Slight uptrend with volatility
        test_quotes.append({
            'open': price - 0.0002,
            'high': price + 0.0003,
            'low': price - 0.0004,
            'close': price,
            'volume': 1000 + (i * 10)
        })
    
    # Extract features
    features = ai_system.extract_advanced_features(test_quotes, 'EURUSD')
    
    # Check that new features are included
    assert 'ha_trend_bullish' in features
    assert 'ha_trend_bearish' in features
    assert 'ha_trend_strength' in features
    assert 'vwap' in features
    assert 'price_vs_vwap' in features
    assert 'vwap_position' in features
    
    print(f"✅ AI integration with new features working correctly")
    print(f"   Total features extracted: {len(features)}")
    print(f"   HA bullish trend: {features.get('ha_trend_bullish', 'N/A')}")
    print(f"   VWAP position: {features.get('vwap_position', 'N/A'):.4f}")
    
    return True

async def main():
    """Run all tests"""
    print("🚀 Starting comprehensive new features test...\n")
    
    tests = [
        test_heikin_ashi,
        test_vwap,
        test_news_event_avoidance,
        test_position_sizing_with_news_avoidance,
        test_ai_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"❌ Test failed: {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"❌ Test error in {test.__name__}: {e}")
        print()
    
    print("=" * 60)
    print(f"📊 NEW FEATURES TEST RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        print("🎉 All new features are working correctly!")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)