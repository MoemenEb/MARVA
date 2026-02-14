from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from evaluation.evaluators.base import BaseEvaluator
from evaluation.util.constants import POS_LABEL


@dataclass
class ScoreMetrics:
    column: str
    support: int
    accuracy: float
    precision: float
    recall: float
    f1: float


class ScoreEvaluator(BaseEvaluator):
    """Compute accuracy, precision, recall, and F1 per evaluation column."""

    def _compute_column(self, col: str) -> ScoreMetrics:
        y_true, y_pred, support = self._prepare_column(col)
        if support == 0:
            return ScoreMetrics(
                column=col, support=0,
                accuracy=0.0, precision=0.0, recall=0.0, f1=0.0,
            )
        return ScoreMetrics(
            column=col,
            support=support,
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
            recall=recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
            f1=f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        )
