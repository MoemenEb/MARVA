from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.plotter.base import BasePlotter
from evaluation.evaluators.duration import DurationAnalyzer


class DurationSummaryPlotter(BasePlotter):
    """Grouped bar chart of mean, median, and total duration per run."""

    def plot_from_analyzer(self, analyzer: DurationAnalyzer,
                           fig_dir: str | Path | None = None) -> Path:
        fig_dir = self._resolve_fig_dir(fig_dir)

        stats = analyzer.summary()
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

        return self._save_figure(fig, fig_dir, "duration_summary.png",
                                 title="Duration Comparison")
