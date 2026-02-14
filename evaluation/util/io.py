from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from evaluation.util.constants import DEFAULT_FIGURES_DIR


def save_summary(df: pd.DataFrame, out_dir: Path, stem: str, suffix: str) -> Path:
    """Write a summary DataFrame to ``{stem}_{suffix}.csv`` inside *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{stem}_{suffix}.csv"
    df.to_csv(out_file, index=False)
    return out_file


def make_fig_dir(fig_dir: str | Path | None = None) -> Path:
    """Create a timestamped subdirectory for figures."""
    base = Path(fig_dir) if fig_dir else DEFAULT_FIGURES_DIR
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def parse_pairs(entries: list[str]) -> dict[str, str]:
    """Parse a list of ``NAME=PATH`` strings into a dict."""
    pairs: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected NAME=PATH format, got: {entry}")
        name, path = entry.split("=", 1)
        pairs[name] = path
    return pairs
