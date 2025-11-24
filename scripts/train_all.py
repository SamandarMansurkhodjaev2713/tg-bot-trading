import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from advanced_trading_ai import AdvancedTradingAI

def main():
    pairs = ['EURUSD','GBPUSD','XAUUSD','BTCUSD','AAPL','USOIL']
    ai = AdvancedTradingAI()
    for tf in ['1h']:
        limit = 4000 if tf=='15m' else 3000 if tf=='1h' else 2000
        print(f"[PIPELINE] timeframe={tf}")
        data = ai.build_unified_dataset(pairs, tf, limit)
        for p in pairs:
            quotes = data.get(p, [])
            if tf=='15m' and len(quotes)<160:
                raw_1m = ai.build_unified_dataset([p], '1m', 6000).get(p, [])
                if raw_1m:
                    quotes = ai._clean_and_align(raw_1m, '15m')
            if not quotes or len(quotes)<180:
                print(f"[SKIP] {p} {tf} insufficient data: {len(quotes)}")
                continue
            res = ai.train_and_compare_models({p: quotes}, p)
            print(f"[MODEL] {p} {tf}: {res}")
    ai.save_models()
    print("[DONE]", datetime.now())

if __name__ == '__main__':
    main()