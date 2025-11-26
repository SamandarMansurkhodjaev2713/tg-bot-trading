import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from itertools import product
from joblib import load
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
    os.makedirs("reports", exist_ok=True)
    grid = {
        "prob_thr": [0.65, 0.7, 0.75],
        "exp_thr": [0.0002, 0.0005, 0.0010],
        "tp_mult": [2.0, 2.5, 3.0],
        "sl_mult": [0.8, 1.0, 1.2],
        "step": [1]
    }
    best_all = {}
    for name in datasets.keys():
        keys = list(grid.keys())
        best = None
        for vals in product(*[grid[k] for k in keys]):
            cfg = dict(zip(keys, vals))
            signals, market_data = generate_signals({name: datasets[name]}, "models/scaler_main.pkl", models, cfg)
            if not signals:
                continue
            engine = AdvancedBacktestingEngine(BacktestConfig())
            bid = f"calib_{name}_{cfg['prob_thr']}_{cfg['exp_thr']}_{cfg['tp_mult']}_{cfg['sl_mult']}_{cfg['step']}"
            md_key = list(market_data.keys())[0]
            metrics = engine.run_backtest(signals, {md_key: market_data[md_key]}, backtest_id=bid)
            score = metrics.profit_factor
            result = {"cfg": cfg, "metrics": {
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "total_return": metrics.total_return,
            }}
            if best is None or score > best["metrics"]["profit_factor"]:
                best = result
        if best:
            best_all[name] = best
    with open("reports/best_signals_by_instrument.json", "w", encoding="utf-8") as f:
        json.dump(best_all, f, ensure_ascii=False, indent=2)
    print(best_all)


if __name__ == "__main__":
    main()
