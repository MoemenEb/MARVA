from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

EXCLUDE_COLUMNS = {"id", "requirement", "recommendations", "duration", "notes", "NOTES"}
POS_LABEL = "FAIL"
LABELS = ["PASS", "FAIL"]


@dataclass
class ColumnMetrics:
    column: str
    support: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int


class ResultsEvaluator:
    """Compare a results CSV against a ground-truth CSV and compute binary
    classification metrics per evaluation column.

    FLAG labels in the results are mapped to FAIL before comparison.
    """

    def __init__(
        self,
        ground_truth: str | Path,
        results: str | Path,
        *,
        id_column: str = "id",
        derive_final_decision: bool = True,
    ) -> None:
        self._gt_path = Path(ground_truth)
        self._res_path = Path(results)
        self._id_column = id_column
        self._derive_final = derive_final_decision

        self._merged: pd.DataFrame | None = None
        self._columns: list[str] = []
        self._metrics: list[ColumnMetrics] | None = None

    def _load(self) -> None:
        gt = pd.read_csv(self._gt_path, dtype=str)
        pred = pd.read_csv(self._res_path, dtype=str)

        id_col = self._id_column
        for label, df in [("Ground truth", gt), ("Results", pred)]:
            if id_col not in df.columns:
                raise ValueError(f"{label} CSV is missing '{id_col}' column")

        gt[id_col] = gt[id_col].fillna("").astype(str).str.strip()
        pred[id_col] = pred[id_col].fillna("").astype(str).str.strip()
        gt = gt[gt[id_col] != ""]
        pred = pred[pred[id_col] != ""]

        self._merged = gt.merge(pred, on=id_col, suffixes=("_gt", "_pred"))

        exclude_lower = {c.lower() for c in EXCLUDE_COLUMNS}
        self._columns = [
            c
            for c in gt.columns
            if c in pred.columns and c.lower() not in exclude_lower
        ]

        if self._derive_final and "final_decision" in pred.columns:
            eval_gt_cols = [c for c in self._columns if c != "final_decision"]
            if eval_gt_cols:
                self._merged["final_decision_gt"] = "PASS"
                for col in eval_gt_cols:
                    gt_vals = self._normalize(self._merged[f"{col}_gt"])
                    self._merged.loc[gt_vals == "FAIL", "final_decision_gt"] = "FAIL"

                self._merged["final_decision_pred"] = self._normalize(
                    self._merged["final_decision"]
                    if "final_decision" in self._merged.columns
                    else self._merged["final_decision_pred"]
                )
                if "final_decision" not in self._columns:
                    self._columns.append("final_decision")

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        s = series.fillna("").astype(str).str.strip().str.upper()
        return s.replace("FLAG", "FAIL")

    def _compute_column(self, col: str) -> ColumnMetrics:
        df = self._merged
        gt_col = f"{col}_gt"
        pred_col = f"{col}_pred"

        y_true = self._normalize(df[gt_col])
        y_pred = self._normalize(df[pred_col])

        mask = (y_true != "") & (y_pred != "")
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        support = len(y_true)
        if support == 0:
            return ColumnMetrics(
                column=col, support=0,
                accuracy=0.0, precision=0.0, recall=0.0, f1=0.0,
                tp=0, fp=0, tn=0, fn=0,
            )

        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

        return ColumnMetrics(
            column=col,
            support=support,
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
            recall=recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
            f1=f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
            tp=int(tp),
            fp=int(fp),
            tn=int(tn),
            fn=int(fn),
        )

    def evaluate(self) -> list[ColumnMetrics]:
        if self._metrics is not None:
            return self._metrics
        if self._merged is None:
            self._load()
        self._metrics = [self._compute_column(col) for col in self._columns]
        return self._metrics

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(m) for m in self.evaluate()])

    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{self._res_path.stem}_metrics.csv"
        self.summary().to_csv(out_file, index=False)
        return out_file
