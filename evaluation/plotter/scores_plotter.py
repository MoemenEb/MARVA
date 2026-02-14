from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.plotter.base import BasePlotter
from evaluation.util.constants import SCORE_METRICS


class ScoresPlotter(BasePlotter):
    """Grouped bar chart: one subplot per metric, columns on x-axis, bars per run."""

    def plot(self, fig_dir: str | Path | None = None) -> Path:
        fig_dir = self._resolve_fig_dir(fig_dir)

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

        return self._save_figure(fig, fig_dir, "scores_comparison.png",
                                 title="Classification Metrics by Run")
