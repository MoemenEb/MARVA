from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


DEFAULT_EXCLUDE = {"id", "requirement", "recommendations", "duration", "notes", "NOTES"}


def _normalize(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def evaluate(ground_truth: Path, results: Path, id_column: str) -> pd.DataFrame:
    gt = pd.read_csv(ground_truth, dtype=str)
    pred = pd.read_csv(results, dtype=str)

    if id_column not in gt.columns:
        raise ValueError(f"Ground truth missing '{id_column}' column: {ground_truth}")
    if id_column not in pred.columns:
        raise ValueError(f"Results missing '{id_column}' column: {results}")

    gt = gt.copy()
    pred = pred.copy()
    gt[id_column] = gt[id_column].fillna("").astype(str).str.strip()
    pred[id_column] = pred[id_column].fillna("").astype(str).str.strip()
    gt = gt[gt[id_column] != ""]
    pred = pred[pred[id_column] != ""]

    df = gt.merge(pred, on=id_column, suffixes=("_gt", "_pred"))

    exclude_lower = {c.lower() for c in DEFAULT_EXCLUDE}
    columns = [c for c in gt.columns if c in pred.columns and c.lower() not in exclude_lower]

    rows: list[dict[str, float | str | int]] = []
    for col in columns:
        y_true = _normalize(df[f"{col}_gt"])
        y_pred = _normalize(df[f"{col}_pred"])
        mask = (y_true != "") & (y_pred != "")
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            rows.append(
                {
                    "column": col,
                    "support": 0,
                    "accuracy": 0.0,
                    "precision_macro": 0.0,
                    "recall_macro": 0.0,
                    "f1_macro": 0.0,
                    "precision_micro": 0.0,
                    "recall_micro": 0.0,
                    "f1_micro": 0.0,
                }
            )
            continue

        accuracy = accuracy_score(y_true, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )

        rows.append(
            {
                "column": col,
                "support": int(len(y_true)),
                "accuracy": float(accuracy),
                "precision_macro": float(p_macro),
                "recall_macro": float(r_macro),
                "f1_macro": float(f1_macro),
                "precision_micro": float(p_micro),
                "recall_micro": float(r_micro),
                "f1_micro": float(f1_micro),
            }
        )

    return pd.DataFrame(rows)


def compare(
    ground_truth: str | Path,
    results: str | Path,
    *,
    out_dir: str | Path | None = None,
    id_column: str = "id",
) -> pd.DataFrame:
    results_path = Path(results)
    table = evaluate(Path(ground_truth), results_path, id_column)

    if out_dir:
        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{results_path.stem}_metrics.csv"
        table.to_csv(out_file, index=False)
        print(f"Wrote: {out_file}")

    return table


if __name__ == "__main__":
    # Edit these paths for each comparison.
    GROUND_TRUTH = Path("data/syntatic/single.csv")
    RESULTS = Path("out/s1_decisions/run_20260210_132621/results.csv")
    OUT_DIR = Path("out/eval_reports")

    metrics = compare(GROUND_TRUTH, RESULTS, out_dir=OUT_DIR)
    print(metrics.to_string(index=False))
