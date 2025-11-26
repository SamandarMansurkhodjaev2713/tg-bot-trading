import os
import sys
from joblib import load
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.data_loader import discover_project_data
from services.signals import generate_signals
from models.finworld import FinWorldAnalyst
from models.flag_trader import FlagTraderAnalyst
from models.finbloom import FinBloom7BAnalyst
from backtesting import AdvancedBacktestingEngine, BacktestConfig


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
        import json
        with open("reports/best_signals.json", "r", encoding="utf-8") as f:
            best = json.load(f)
            cfg = best.get("cfg")
    except Exception:
        cfg = {"prob_thr": 0.7, "exp_thr": 0.0005, "tp_mult": 2.5, "sl_mult": 1.2, "step": 1}
    per = None
    try:
        import json
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
    engine = AdvancedBacktestingEngine(BacktestConfig())
    metrics = engine.run_backtest(signals, market_data)
    print({
        "total_trades": metrics.total_trades,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "total_return": metrics.total_return,
        "sharpe_ratio": metrics.sharpe_ratio,
    })


if __name__ == "__main__":
    main()
