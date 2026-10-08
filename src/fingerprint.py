"""Dataset fingerprint (SHA-256) so anyone can verify they use the same data."""
import hashlib

import pandas as pd


def data_fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
    df = X.copy()
    df["target"] = y.values
    return hashlib.sha256(df.to_csv(index=False, float_format="%.10g").encode()).hexdigest()
