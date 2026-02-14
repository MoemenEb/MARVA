from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.util.stats import remove_outliers_iqr


class DurationAnalyzer:
    """Compare per-requirement durations across runs.

    Loads the ``duration`` column from results CSVs, removes outliers via
    the IQR method, and produces summary statistics.
    """

    def __init__(self, runs: dict[str, str | Path]) -> None:
        self._raw: dict[str, pd.Series] = {}
        self._clean: dict[str, pd.Series] = {}
        for name, path in runs.items():
            df = pd.read_csv(path, dtype={"duration": float})
            durations = pd.to_numeric(df["duration"], errors="coerce").dropna()
            self._raw[name] = durations
            self._clean[name] = remove_outliers_iqr(durations)

    @property
    def run_names(self) -> list[str]:
        return list(self._raw.keys())

    @property
    def raw(self) -> dict[str, pd.Series]:
        return dict(self._raw)

    @property
    def clean(self) -> dict[str, pd.Series]:
        return dict(self._clean)

    def summary(self) -> pd.DataFrame:
        rows = []
        for name in self.run_names:
            raw = self._raw[name]
            clean = self._clean[name]
            rows.append(
                {
                    "run": name,
                    "count": len(raw),
                    "outliers_removed": len(raw) - len(clean),
                    "mean": clean.mean(),
                    "median": clean.median(),
                    "std": clean.std(),
                    "min": clean.min(),
                    "max": clean.max(),
                    "total": raw.sum(),
                    "total_clean": clean.sum(),
                }
            )
        return pd.DataFrame(rows)
