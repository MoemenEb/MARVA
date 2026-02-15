from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.evaluators.score_evaluator import ScoreEvaluator
from evaluation.evaluators.confusion_evaluator import ConfusionEvaluator
from evaluation.evaluators.duration import DurationAnalyzer
from evaluation.evaluators.cross_run_analyzer import CrossRunAnalyzer
from evaluation.util.constants import GROUND_TRUTH_MAP, DEFAULT_OUT_DIR, EVAL_MODES
from evaluation.util.io import save_summary


def main(results: str, mode: str, eval_mode: str, out_dir: str | None,
         duration: bool = False) -> None:
    gt_path = GROUND_TRUTH_MAP.get(mode)
    if gt_path is None:
        raise ValueError(f"No ground truth configured for mode '{mode}'")

    save_dir = DEFAULT_OUT_DIR / (out_dir or "")
    res_stem = Path(results).stem
    scores_df = None
    confusion_df = None

    if eval_mode in ("scores", "both"):
        evaluator = ScoreEvaluator(gt_path, results)
        scores_df = evaluator.summary()
        print("=== Score Metrics ===")
        print(scores_df.to_string(index=False))
        saved = save_summary(scores_df, save_dir, res_stem, "scores")
        print(f"Saved to: {saved}\n")

    if eval_mode in ("confusion", "both"):
        evaluator = ConfusionEvaluator(gt_path, results)
        confusion_df = evaluator.summary()
        print("=== Confusion Metrics ===")
        print(confusion_df.to_string(index=False))
        saved = save_summary(confusion_df, save_dir, res_stem, "confusion")
        print(f"Saved to: {saved}\n")

    if eval_mode == "both" and scores_df is not None and confusion_df is not None:
        combined = scores_df.merge(
            confusion_df.drop(columns=["support"]), on="column"
        )
        saved = save_summary(combined, save_dir, res_stem, "metrics")
        print(f"Combined saved to: {saved}\n")

    if duration:
        analyzer = DurationAnalyzer({res_stem: results})
        duration_df = analyzer.summary()
        print("=== Duration Stats ===")
        print(duration_df.to_string(index=False))
        saved = save_summary(duration_df, save_dir, res_stem, "duration")
        print(f"Saved to: {saved}")


def run_stats(
    results_dir: str | None,
    arch_pairs: list[list[str]] | None,
    out_dir: str | None,
) -> None:
    """Run cross-run statistical analysis across architectures."""
    save_dir = DEFAULT_OUT_DIR / (out_dir or "")

    if arch_pairs:
        arch_map: dict[str, list[str]] = {}
        for pair_group in arch_pairs:
            for entry in pair_group:
                if "=" not in entry:
                    raise ValueError(f"Expected NAME=PATH, got: {entry}")
                name, path = entry.split("=", 1)
                arch_map.setdefault(name, []).append(path)
        analyzer = CrossRunAnalyzer(arch_map)
    else:
        analyzer = CrossRunAnalyzer.discover(results_dir or DEFAULT_OUT_DIR)

    # 1. Descriptive statistics
    desc_df = analyzer.descriptive_summary()
    print("=== Descriptive Statistics ===")
    print(desc_df.to_string(index=False))
    saved = save_summary(desc_df, save_dir, "cross_run", "descriptive")
    print(f"Saved to: {saved}\n")

    # 2. Paired t-tests
    ttest_df = analyzer.paired_ttest_summary()
    print("=== Paired T-Tests ===")
    print(ttest_df.to_string(index=False))
    saved = save_summary(ttest_df, save_dir, "cross_run", "ttest")
    print(f"Saved to: {saved}\n")

    # 3. Effect sizes (Cohen's d)
    effect_df = analyzer.effect_size_summary()
    print("=== Effect Sizes (Cohen's d) ===")
    print(effect_df.to_string(index=False))
    saved = save_summary(effect_df, save_dir, "cross_run", "effect_size")
    print(f"Saved to: {saved}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate results against ground truth")
    parser.add_argument(
        "--results",
        default=None,
        help="Path to results.csv from a run",
    )
    parser.add_argument(
        "--mode",
        default="single",
        choices=["single"],
        help="Evaluation mode (default: single)",
    )
    parser.add_argument(
        "--eval-mode",
        default="both",
        choices=EVAL_MODES,
        help="Which metrics to compute: scores, confusion, or both (default: both)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=f"Output directory for metrics CSV (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--duration",
        action="store_true",
        help="Also compute duration statistics",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Run cross-run statistical analysis across architectures",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Root directory for auto-discovering run folders (default: results/)",
    )
    parser.add_argument(
        "--arch",
        nargs="+",
        metavar="NAME=PATH",
        action="append",
        help=(
            "Architecture runs as NAME=CSV_PATH pairs. "
            "Repeat --arch for each run."
        ),
    )
    args = parser.parse_args()

    if args.stats:
        run_stats(args.results_dir, args.arch, args.out_dir)
    else:
        if not args.results:
            parser.error("--results is required when not using --stats")
        main(args.results, args.mode, args.eval_mode, args.out_dir, args.duration)
