from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.util.constants import DEFAULT_FIGURES_DIR
from evaluation.util.io import make_fig_dir, parse_pairs
from evaluation.plotter.scores_plotter import ScoresPlotter
from evaluation.plotter.confusion_plotter import ConfusionPlotter
from evaluation.plotter.duration_box_plotter import DurationBoxPlotter
from evaluation.plotter.duration_summary_plotter import DurationSummaryPlotter
from evaluation.evaluators.duration import DurationAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot evaluation metrics")
    parser.add_argument(
        "--metrics",
        nargs="+",
        metavar="NAME=PATH",
        help="Metrics CSVs as name=path pairs (e.g. s1=results/s1/results_metrics.csv)",
    )
    parser.add_argument(
        "--duration",
        nargs="+",
        metavar="NAME=PATH",
        help="Results CSVs for duration plots as name=path pairs",
    )
    parser.add_argument(
        "--fig-dir",
        default=None,
        help=f"Base figures directory (default: {DEFAULT_FIGURES_DIR})",
    )
    args = parser.parse_args()

    if not args.metrics and not args.duration:
        parser.error("At least one of --metrics or --duration is required")

    fig_dir = Path(args.fig_dir) if args.fig_dir else make_fig_dir()

    if args.metrics:
        pairs = parse_pairs(args.metrics)

        path = ScoresPlotter(pairs).plot(fig_dir)
        print(f"Saved: {path}")

        path = ConfusionPlotter(pairs).plot(fig_dir)
        print(f"Saved: {path}")

    if args.duration:
        pairs = parse_pairs(args.duration)
        analyzer = DurationAnalyzer(pairs)

        path = DurationBoxPlotter().plot_from_analyzer(analyzer, fig_dir)
        print(f"Saved: {path}")

        path = DurationSummaryPlotter().plot_from_analyzer(analyzer, fig_dir)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
