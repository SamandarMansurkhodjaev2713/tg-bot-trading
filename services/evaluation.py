import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, mean_squared_error

def evaluate(models, X_val, y_val_cls, X_train, data):
    metrics = {}
    for name, m in models.items():
        try:
            if hasattr(m, "predict"):
                y_pred = m.predict(X_val)
                if y_pred.dtype == np.float64 or y_pred.dtype == np.float32:
                    mae = mean_absolute_error(data["future_return_5"].iloc[-len(y_pred):].values, y_pred)
                    rmse = mean_squared_error(data["future_return_5"].iloc[-len(y_pred):].values, y_pred, squared=False)
                    metrics[name] = {"mae": float(mae), "rmse": float(rmse)}
                else:
                    acc = accuracy_score(y_val_cls, y_pred)
                    prec = precision_score(y_val_cls, y_pred, zero_division=0)
                    rec = recall_score(y_val_cls, y_pred, zero_division=0)
                    metrics[name] = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec)}
        except Exception:
            continue
    return metrics

def fine_tune(models, metrics, X_train, y_train_cls, data):
    updated = dict(models)
    acc_threshold = 0.60
    rmse_threshold = 0.015
    if metrics.get("FinWorld", {}).get("accuracy", 1.0) < acc_threshold:
        m = models["FinWorld"].__class__()
        m.fit(X_train, y_train_cls)
        m.save("models/finworld_ft.pkl")
        updated["FinWorld"] = m
    if metrics.get("FLAG-Trader", {}).get("accuracy", 1.0) < acc_threshold:
        m = models["FLAG-Trader"].__class__()
        m.fit(X_train, y_train_cls)
        m.save("models/flag_trader_ft.pkl")
        updated["FLAG-Trader"] = m
    if metrics.get("FinBloom-7B", {}).get("rmse", 0.0) > rmse_threshold:
        y_train_reg = data["future_return_5"].values[: len(X_train)]
        m = models["FinBloom-7B"].__class__()
        m.fit(X_train, y_train_reg)
        m.save("models/finbloom_7b_ft.pkl")
        updated["FinBloom-7B"] = m
    X_val = X_train[-max(1, int(0.2 * len(X_train))):]
    y_val_cls = y_train_cls[-len(X_val):]
    new_metrics = evaluate(updated, X_val, y_val_cls, X_train, data)
    return updated, new_metrics
