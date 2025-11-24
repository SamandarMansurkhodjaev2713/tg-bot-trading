import json
import os
import sys
from datetime import datetime

# Ensure project root is on PYTHONPATH
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from advanced_trading_ai import AdvancedTradingAI

def main():
    ai = AdvancedTradingAI()
    pair = 'EURUSD'
    results = {}
    for tf in ['15m','1h','1d']:
        limit = 4000 if tf == '15m' else 3000 if tf == '1h' else 2000
        data = ai.build_unified_dataset([pair], tf, limit)
        quotes = data.get(pair, [])
        if tf == '15m' and len(quotes) < 160:
            raw_1m = ai.build_unified_dataset([pair], '1m', 6000).get(pair, [])
            if raw_1m:
                quotes = ai._clean_and_align(raw_1m, '15m')
        print(f"[DATA] {pair} {tf}: quotes={len(quotes)}")
        res = ai.train_improved({pair: quotes}, tf)
        results[tf] = res
        print(f"[TRAIN] {tf}: {res}")
    ai.save_models()
    print("[DONE] Models saved at", datetime.now())

if __name__ == '__main__':
    main()