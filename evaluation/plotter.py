from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCORE_METRICS = ["accuracy", "precision", "recall", "f1"]
DEFAULT_FIGURES_DIR = Path("figures")


class MetricsPlotter:
    """Load one or more metrics CSVs and produce comparison plots.

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

    def _make_fig_dir(self, fig_dir: str | Path | None) -> Path:
        base = Path(fig_dir) if fig_dir else DEFAULT_FIGURES_DIR
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = base / ts
        out.mkdir(parents=True, exist_ok=True)
        return out

    def plot_scores(self, fig_dir: str | Path | None = None, *, _resolved: bool = False) -> Path:
        """Grouped bar chart: one subplot per metric, columns on x-axis, bars per run."""
        fig_dir = Path(fig_dir) if _resolved and fig_dir else self._make_fig_dir(fig_dir)

        runs = self._runs
        run_names = self.run_names
        columns = runs[run_names[0]]["column"].tolist()
        n_cols = len(columns)
        n_runs = len(run_names)

        fig, axes = plt.subplots(1, len(SCORE_METRICS), figsize=(5 * len(SCORE_METRICS), 5))
        if len(SCORE_METRICS) == 1:
            axes = [axes]

        x = np.arange(n_cols)
        width = 0.8 / n_runs

        for ax, metric in zip(axes, SCORE_METRICS):
            for i, run in enumerate(run_names):
                vals = runs[run].set_index("column").reindex(columns)[metric].fillna(0).values
                offset = (i - n_runs / 2 + 0.5) * width
                bars = ax.bar(x + offset, vals, width, label=run)
                for bar, v in zip(bars, vals):
                    if v > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{v:.2f}",
                            ha="center", va="bottom", fontsize=7,
                        )
            ax.set_title(metric.capitalize(), fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(columns, rotation=30, ha="right")
            ax.set_ylim(0, 1.15)
            ax.legend(fontsize=8)

        fig.suptitle("Classification Metrics by Run", fontweight="bold", fontsize=13)
        fig.tight_layout()
        path = fig_dir / "scores_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_confusion_matrices(self, fig_dir: str | Path | None = None, *, _resolved: bool = False) -> Path:
        """Grid of 2x2 confusion-matrix heatmaps: rows = columns, cols = runs."""
        fig_dir = Path(fig_dir) if _resolved and fig_dir else self._make_fig_dir(fig_dir)

        runs = self._runs
        run_names = self.run_names
        columns = runs[run_names[0]]["column"].tolist()

        n_rows = len(columns)
        n_cols = len(run_names)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.5 * n_cols, 3 * n_rows),
            squeeze=False,
        )

        for r, col in enumerate(columns):
            for c, run in enumerate(run_names):
                ax = axes[r][c]
                row = runs[run].set_index("column").loc[col]
                cm = np.array([[int(row["tn"]), int(row["fp"])],
                               [int(row["fn"]), int(row["tp"])]])

                ax.imshow(cm, cmap="Blues", vmin=0)
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(cm[i, j]),
                                ha="center", va="center", fontsize=14, fontweight="bold")
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["PASS", "FAIL"])
                ax.set_yticklabels(["PASS", "FAIL"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")

                if r == 0:
                    ax.set_title(run, fontweight="bold")
                if c == 0:
                    ax.annotate(
                        col, xy=(-0.4, 0.5), xycoords="axes fraction",
                        fontsize=11, fontweight="bold", ha="right", va="center", rotation=90,
                    )

        fig.suptitle("Confusion Matrices", fontweight="bold", fontsize=13)
        fig.tight_layout()
        path = fig_dir / "confusion_matrices.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def save_all(self, fig_dir: str | Path | None = None) -> list[Path]:
        out = self._make_fig_dir(fig_dir)
        return [
            self.plot_scores(out, _resolved=True),
            self.plot_confusion_matrices(out, _resolved=True),
        ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot evaluation metrics")
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="Metrics CSVs as name=path pairs (e.g. s1=results/s1/results_metrics.csv)",
    )
    parser.add_argument(
        "--fig-dir",
        default=None,
        help=f"Base figures directory (default: {DEFAULT_FIGURES_DIR})",
    )
    args = parser.parse_args()

    pairs: dict[str, str] = {}
    for entry in args.metrics:
        if "=" not in entry:
            parser.error(f"Expected NAME=PATH format, got: {entry}")
        name, path = entry.split("=", 1)
        pairs[name] = path

    plotter = MetricsPlotter(pairs)
    paths = plotter.save_all(args.fig_dir)
    for p in paths:
        print(f"Saved: {p}")
