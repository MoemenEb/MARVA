from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import confusion_matrix

from evaluation.evaluators.base import BaseEvaluator
from evaluation.util.constants import LABELS


@dataclass
class ConfusionMetrics:
    column: str
    support: int
    tp: int
    fp: int
    tn: int
    fn: int


class ConfusionEvaluator(BaseEvaluator):
    """Compute TP, TN, FP, FN per evaluation column."""

    def _compute_column(self, col: str) -> ConfusionMetrics:
        y_true, y_pred, support = self._prepare_column(col)
        if support == 0:
            return ConfusionMetrics(
                column=col, support=0,
                tp=0, fp=0, tn=0, fn=0,
            )
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        return ConfusionMetrics(
            column=col,
            support=support,
            tp=int(tp),
            fp=int(fp),
            tn=int(tn),
            fn=int(fn),
        )
