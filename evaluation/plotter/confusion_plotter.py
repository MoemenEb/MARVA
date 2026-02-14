from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.plotter.base import BasePlotter


class ConfusionPlotter(BasePlotter):
    """Grid of 2x2 confusion-matrix heatmaps: rows = columns, cols = runs."""

    def plot(self, fig_dir: str | Path | None = None, *, _resolved: bool = False) -> Path:
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
