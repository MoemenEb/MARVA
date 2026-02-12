import argparse
from pathlib import Path

from evaluation.evaluator import ResultsEvaluator

GROUND_TRUTH_DIR = Path("data/syntatic")
DEFAULT_OUT_DIR = Path("results")

GROUND_TRUTH_MAP = {
    "single": GROUND_TRUTH_DIR / "single_2.csv",
}


def main(results: str, mode: str, out_dir: str | None) -> None:
    gt_path = GROUND_TRUTH_MAP.get(mode)
    if gt_path is None:
        raise ValueError(f"No ground truth configured for mode '{mode}'")

    evaluator = ResultsEvaluator(gt_path, results)
    df = evaluator.summary()
    print(df.to_string(index=False))

    save_dir = DEFAULT_OUT_DIR / (out_dir or "")
    saved = evaluator.save(save_dir)
    print(f"\nSaved to: {saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate results against ground truth")
    parser.add_argument(
        "--results",
        required=True,
        help="Path to results.csv from a run",
    )
    parser.add_argument(
        "--mode",
        default="single",
        choices=["single"],
        help="Evaluation mode (default: single)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=f"Output directory for metrics CSV (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()
    main(args.results, args.mode, args.out_dir)
