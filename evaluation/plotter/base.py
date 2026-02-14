from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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

    # ------------------------------------------------------------------
    # Figure helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_fig_dir(fig_dir: str | Path | None = None) -> Path:
        if isinstance(fig_dir, Path):
            fig_dir.mkdir(parents=True, exist_ok=True)
            return fig_dir
        return make_fig_dir(fig_dir)

    @staticmethod
    def _save_figure(fig: plt.Figure, fig_dir: Path, filename: str,
                     title: str = "") -> Path:
        if title:
            fig.suptitle(title, fontweight="bold", fontsize=13)
        fig.tight_layout()
        path = fig_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
