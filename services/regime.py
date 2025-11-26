import pandas as pd
from utils.indicators import adx, boll_bw

def classify_regime(df: pd.DataFrame) -> pd.Series:
    a = adx(df).fillna(0)
    b = boll_bw(df).fillna(0)
    reg = []
    for i in range(len(df)):
        ai = float(a.iloc[i]) if i < len(a) else 0.0
        bi = float(b.iloc[i]) if i < len(b) else 0.0
        if ai >= 20 and bi < 0.04:
            reg.append("trend")
        elif bi >= 0.08:
            reg.append("volatile")
        else:
            reg.append("chop")
    return pd.Series(reg, index=df.index)
