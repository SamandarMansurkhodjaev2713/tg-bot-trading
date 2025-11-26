import os
import sys
import json
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from joblib import load
from services.data_loader import discover_project_data
from services.signals import generate_signals
from models.finworld import FinWorldAnalyst
from models.flag_trader import FlagTraderAnalyst
from models.finbloom import FinBloom7BAnalyst


def load_models():
    fw = FinWorldAnalyst()
    ft = FlagTraderAnalyst()
    fb = FinBloom7BAnalyst()
    fw.model = load("models/finworld.pkl")
    ft.model = load("models/flag_trader.pkl")
    fb.model = load("models/finbloom_7b.pkl")
    return {"FinWorld": fw, "FLAG-Trader": ft, "FinBloom-7B": fb}


def main():
    datasets = discover_project_data()
    models = load_models()
    cfg = None
    try:
        with open("reports/best_signals.json", "r", encoding="utf-8") as f:
            best = json.load(f)
            cfg = best.get("cfg")
    except Exception:
        cfg = {"prob_thr": 0.7, "exp_thr": 0.0005, "tp_mult": 2.5, "sl_mult": 1.2, "step": 1}
    per = None
    try:
        with open("reports/best_signals_by_instrument.json", "r", encoding="utf-8") as f:
            per = json.load(f)
    except Exception:
        per = None
    per_map = None
    if per:
        per_map = {}
        for k, v in per.items():
            import os
            base = os.path.splitext(os.path.basename(k))[0].upper()
            per_map[base] = v.get("cfg", v)
    signals, market_data = generate_signals(datasets, "models/scaler_main.pkl", models, cfg, per_map)
    os.makedirs("signals", exist_ok=True)
    with open("signals/signals.json", "w", encoding="utf-8") as fj:
        json.dump(signals, fj, ensure_ascii=False, indent=2)
    with open("signals/signals.csv", "w", newline="", encoding="utf-8") as fc:
        w = csv.DictWriter(fc, fieldnames=list(signals[0].keys()) if signals else ["timestamp","pair","direction","entry_price","stop_loss","take_profit","confidence","market_regime"])
        w.writeheader()
        for s in signals:
            w.writerow(s)
    print({"saved": len(signals), "json": "signals/signals.json", "csv": "signals/signals.csv"})


if __name__ == "__main__":
    main()
