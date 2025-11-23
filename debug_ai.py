#!/usr/bin/env python3
"""Debug script for AI Models test"""

from datetime import datetime, timedelta
from advanced_trading_ai import AdvancedTradingAI

def debug_ai_models():
    """Debug AI models test"""
    print("🧪 Debugging AI Models...")
    
    try:
        # Initialize AI system
        ai_system = AdvancedTradingAI()
        print("✅ Advanced AI system initialized")
        
        # Create test data
        test_quotes = []
        base_price = 1.1000
        for i in range(100):
            quote = {
                'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
                'open': base_price + (i * 0.0001),
                'high': base_price + (i * 0.0002),
                'low': base_price - (i * 0.0001),
                'close': base_price + (i * 0.00015),
                'volume': 1000 + i * 10
            }
            test_quotes.append(quote)
        
        print(f"Created {len(test_quotes)} test quotes")
        
        # Test feature extraction
        print("Testing feature extraction...")
        features = ai_system.extract_advanced_features(test_quotes, 'EURUSD')
        print(f"✅ Feature extraction: {len(features)} features extracted")
        print(f"Sample features: {list(features.keys())[:10]}")
        
        # Test prediction step by step
        print("Testing prediction...")
        
        # Try the predict method directly
        try:
            signal, probability, confidence = ai_system.predict(features)
            print(f"Direct predict: signal={signal}, probability={probability}, confidence={confidence}")
        except Exception as e:
            print(f"Direct predict error: {e}")
            import traceback
            traceback.print_exc()
        
        # Try predict_with_confidence
        try:
            prediction = ai_system.predict_with_confidence(test_quotes)
            print(f"predict_with_confidence: {prediction}")
        except Exception as e:
            print(f"predict_with_confidence error: {e}")
            import traceback
            traceback.print_exc()
        
        print("Debug completed!")
        
    except Exception as e:
        print(f"Debug error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ai_models()