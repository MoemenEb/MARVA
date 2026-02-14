from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from evaluation.util.constants import (
    EXCLUDE_COLUMNS,
    FLAG_LABEL,
    ATOMICITY_COL,
)


class BaseEvaluator:
    """Load and merge a results CSV against a ground-truth CSV.

    FLAG labels in the results are mapped to FAIL before comparison.
    Subclasses must implement ``_compute_column(col)`` to return a
    dataclass instance with the per-column metrics.
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
        self._metrics: list | None = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

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
            self._derive_final_decision()

    def _derive_final_decision(self) -> None:
        eval_gt_cols = [c for c in self._columns if c != "final_decision"]
        if not eval_gt_cols:
            return

        self._merged["final_decision_gt"] = "PASS"
        for col in eval_gt_cols:
            gt_vals = self._normalize(self._merged[f"{col}_gt"])
            if col == ATOMICITY_COL:
                self._merged.loc[gt_vals == "FAIL", "final_decision_gt"] = "FAIL"
            else:
                mask = (gt_vals == "FAIL") & (
                    self._merged["final_decision_gt"] != "FAIL"
                )
                self._merged.loc[mask, "final_decision_gt"] = FLAG_LABEL

        self._merged["final_decision_gt"] = self._merged[
            "final_decision_gt"
        ].map(lambda x: "FAIL" if x == FLAG_LABEL else x)

        self._merged["final_decision_pred"] = self._normalize(
            self._merged["final_decision"]
            if "final_decision" in self._merged.columns
            else self._merged["final_decision_pred"]
        )
        if "final_decision" not in self._columns:
            self._columns.append("final_decision")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        s = series.fillna("").astype(str).str.strip().str.upper()
        return s.map(lambda x: x if x in ("PASS", "") else "FAIL")

    def _prepare_column(self, col: str) -> tuple[pd.Series, pd.Series, int]:
        """Return (y_true, y_pred, support) for a column, loading data if needed."""
        if self._merged is None:
            self._load()
        df = self._merged
        y_true = self._normalize(df[f"{col}_gt"])
        y_pred = self._normalize(df[f"{col}_pred"])
        mask = (y_true != "") & (y_pred != "")
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        return y_true, y_pred, len(y_true)

    @property
    def columns(self) -> list[str]:
        if self._merged is None:
            self._load()
        return list(self._columns)

    # ------------------------------------------------------------------
    # Template methods — subclasses implement _compute_column()
    # ------------------------------------------------------------------

    def _compute_column(self, col: str):
        raise NotImplementedError

    def evaluate(self) -> list:
        if self._metrics is not None:
            return self._metrics
        if self._merged is None:
            self._load()
        self._metrics = [self._compute_column(col) for col in self._columns]
        return self._metrics

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(m) for m in self.evaluate()])
