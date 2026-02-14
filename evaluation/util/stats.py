from __future__ import annotations

import pandas as pd


def remove_outliers_iqr(s: pd.Series) -> pd.Series:
    """Remove outliers from a numeric series using the IQR method."""
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return s[(s >= lower) & (s <= upper)]
