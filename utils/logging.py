import os
import json
from datetime import datetime

def save_logs(metrics):
    os.makedirs("reports", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    lines = [f"{k}: {v}" for k, v in metrics.items()]
    with open("reports/report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open("logs/train.log", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] TRAINED: {lines}\n")
    with open("logs/test.log", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] EVAL: {lines}\n")
