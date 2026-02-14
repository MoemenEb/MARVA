from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.util.io import make_fig_dir


class BasePlotter:
    """Load one or more metrics CSVs for plotting.

    Metrics can be supplied explicitly as name->path pairs, or auto-discovered
    from a directory.
    """

    def __init__(
        self,
        metrics: dict[str, str | Path] | str | Path | None = None,
    ) -> None:
        self._runs: dict[str, pd.DataFrame] = {}

        if metrics is None:
            return
        if isinstance(metrics, dict):
            for name, path in metrics.items():
                self._runs[name] = pd.read_csv(path)
        else:
            self._scan_dir(Path(metrics))

        if not self._runs:
            raise FileNotFoundError(f"No metrics loaded from {metrics}")

    def _scan_dir(self, root: Path) -> None:
        for csv in sorted(root.rglob("*_metrics.csv")):
            run_name = csv.parent.name if csv.parent != root else csv.stem
            self._runs[run_name] = pd.read_csv(csv)

    def add(self, name: str, path: str | Path) -> None:
        self._runs[name] = pd.read_csv(path)

    @property
    def run_names(self) -> list[str]:
        return list(self._runs.keys())

    @staticmethod
    def _make_fig_dir(fig_dir: str | Path | None = None) -> Path:
        return make_fig_dir(fig_dir)
