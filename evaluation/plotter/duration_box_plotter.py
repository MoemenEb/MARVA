from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from evaluation.plotter.base import BasePlotter
from evaluation.evaluators.duration import DurationAnalyzer


class DurationBoxPlotter(BasePlotter):
    """Box plot of per-requirement durations (after outlier removal)."""

    def plot_from_analyzer(self, analyzer: DurationAnalyzer,
                           fig_dir: str | Path | None = None) -> Path:
        fig_dir = self._resolve_fig_dir(fig_dir)

        run_names = analyzer.run_names
        clean = analyzer.clean
        data = [clean[r].values for r in run_names]

        fig, ax = plt.subplots(figsize=(max(4, 2.5 * len(run_names)), 5))
        bp = ax.boxplot(data, tick_labels=run_names, patch_artist=True)
        colors = plt.cm.tab10.colors
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_ylabel("Duration (s)")
        ax.set_title("Per-Requirement Duration (outliers removed)", fontweight="bold")

        return self._save_figure(fig, fig_dir, "duration_boxplot.png")
