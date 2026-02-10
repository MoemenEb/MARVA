from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_FIGURES_DIR = Path("figures")


class DurationAnalyzer:
    """Compare per-requirement durations across runs.

    Loads the ``duration`` column from results CSVs, removes outliers via
    the IQR method, and produces summary statistics and comparison plots.
    """

    def __init__(self, runs: dict[str, str | Path]) -> None:
        self._raw: dict[str, pd.Series] = {}
        self._clean: dict[str, pd.Series] = {}
        for name, path in runs.items():
            df = pd.read_csv(path, dtype={"duration": float})
            durations = pd.to_numeric(df["duration"], errors="coerce").dropna()
            self._raw[name] = durations
            self._clean[name] = self._remove_outliers(durations)

    @staticmethod
    def _remove_outliers(s: pd.Series) -> pd.Series:
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return s[(s >= lower) & (s <= upper)]

    @property
    def run_names(self) -> list[str]:
        return list(self._raw.keys())

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

    def _make_fig_dir(self, fig_dir: str | Path | None) -> Path:
        base = Path(fig_dir) if fig_dir else DEFAULT_FIGURES_DIR
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = base / ts
        out.mkdir(parents=True, exist_ok=True)
        return out

    def plot_box(self, fig_dir: str | Path | None = None, *, _resolved: bool = False) -> Path:
        """Box plot of per-requirement durations (after outlier removal)."""
        fig_dir = Path(fig_dir) if _resolved and fig_dir else self._make_fig_dir(fig_dir)

        data = [self._clean[r].values for r in self.run_names]
        fig, ax = plt.subplots(figsize=(max(4, 2.5 * len(self.run_names)), 5))
        bp = ax.boxplot(data, tick_labels=self.run_names, patch_artist=True)
        colors = plt.cm.tab10.colors
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_ylabel("Duration (s)")
        ax.set_title("Per-Requirement Duration (outliers removed)", fontweight="bold")
        fig.tight_layout()
        path = fig_dir / "duration_boxplot.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_summary_bars(self, fig_dir: str | Path | None = None, *, _resolved: bool = False) -> Path:
        """Grouped bar chart of mean, median, and total duration per run."""
        fig_dir = Path(fig_dir) if _resolved and fig_dir else self._make_fig_dir(fig_dir)

        stats = self.summary()
        run_names = stats["run"].tolist()
        x = np.arange(len(run_names))

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Mean & median (per-requirement, cleaned)
        ax = axes[0]
        w = 0.35
        bars_mean = ax.bar(x - w / 2, stats["mean"], w, label="Mean")
        bars_med = ax.bar(x + w / 2, stats["median"], w, label="Median")
        for bars in [bars_mean, bars_med]:
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=8,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(run_names)
        ax.set_ylabel("Seconds")
        ax.set_title("Avg Duration per Requirement", fontweight="bold")
        ax.legend()

        # Total duration
        ax = axes[1]
        total_mins = stats["total"] / 60
        bars_total = ax.bar(x, total_mins, 0.5, color=plt.cm.tab10.colors[:len(run_names)])
        for bar in bars_total:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{bar.get_height():.1f}m",
                ha="center", va="bottom", fontsize=8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(run_names)
        ax.set_ylabel("Minutes")
        ax.set_title("Total Run Duration", fontweight="bold")

        fig.suptitle("Duration Comparison", fontweight="bold", fontsize=13)
        fig.tight_layout()
        path = fig_dir / "duration_summary.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def save_all(self, fig_dir: str | Path | None = None) -> list[Path]:
        out = self._make_fig_dir(fig_dir)
        stats_path = "results/duration_stats.csv"
        self.summary().to_csv(stats_path, index=False)
        return [
            stats_path,
            self.plot_box(out, _resolved=True),
            self.plot_summary_bars(out, _resolved=True),
        ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze and plot run durations")
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="Results CSVs as name=path pairs (e.g. s1=out/s1_decisions/run_.../results.csv)",
    )
    parser.add_argument(
        "--fig-dir",
        default=None,
        help=f"Base figures directory (default: {DEFAULT_FIGURES_DIR})",
    )
    args = parser.parse_args()

    pairs: dict[str, str] = {}
    for entry in args.results:
        if "=" not in entry:
            parser.error(f"Expected NAME=PATH format, got: {entry}")
        name, path = entry.split("=", 1)
        pairs[name] = path

    analyzer = DurationAnalyzer(pairs)
    print(analyzer.summary().to_string(index=False))
    print()
    paths = analyzer.save_all(args.fig_dir)
    for p in paths:
        print(f"Saved: {p}")
