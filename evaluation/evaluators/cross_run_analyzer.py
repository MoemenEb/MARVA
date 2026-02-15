"""Cross-run statistical analysis across architectures."""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from evaluation.util.constants import STAT_METRICS, DEFAULT_OUT_DIR


class CrossRunAnalyzer:
    """Compare metric distributions across multiple runs grouped by architecture.

    Each architecture (e.g. s1, s2, s3) has N runs, each producing a
    ``results_metrics.csv`` with columns: column, support, accuracy,
    precision, recall, f1, tp, fp, tn, fn.

    This analyzer computes:
    - Descriptive statistics per architecture per dimension per metric
    - Paired t-tests between architecture pairs
    - Cohen's d effect sizes between architecture pairs
    - 95% confidence intervals per architecture
    """

    _PREFIX_RE = re.compile(r"^(s\d+)_\d{8}_\d{6}$")

    def __init__(
        self,
        architectures: dict[str, list[str | Path]],
        *,
        metrics: list[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        architectures : dict[str, list[str | Path]]
            Mapping of architecture name (e.g. "s1") to a list of paths
            pointing to results_metrics.csv files.
        metrics : list[str] | None
            Metric columns to analyze. Defaults to STAT_METRICS
            (precision, recall, f1).
        """
        self._metrics = metrics or list(STAT_METRICS)
        self._arch_names: list[str] = sorted(architectures.keys())
        self._data: dict[str, list[pd.DataFrame]] = {}
        for arch, paths in architectures.items():
            self._data[arch] = [pd.read_csv(p) for p in paths]

    @classmethod
    def discover(
        cls,
        results_dir: str | Path = DEFAULT_OUT_DIR,
        *,
        metrics: list[str] | None = None,
        csv_name: str = "results_metrics.csv",
    ) -> CrossRunAnalyzer:
        """Auto-discover runs from a results directory.

        Scans for subdirectories matching ``{prefix}_{YYYYMMDD}_{HHMMSS}``
        and groups them by architecture prefix.
        """
        root = Path(results_dir)
        arch_map: dict[str, list[Path]] = {}
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            match = cls._PREFIX_RE.match(child.name)
            if not match:
                continue
            prefix = match.group(1)
            csv_path = child / csv_name
            if csv_path.exists():
                arch_map.setdefault(prefix, []).append(csv_path)
        if not arch_map:
            raise FileNotFoundError(
                f"No run directories found in {root}"
            )
        return cls(arch_map, metrics=metrics)

    @property
    def arch_names(self) -> list[str]:
        return list(self._arch_names)

    @property
    def dimensions(self) -> list[str]:
        """Return the dimension names from the first run."""
        first_arch = self._arch_names[0]
        return self._data[first_arch][0]["column"].tolist()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_metric_values(
        self, arch: str, dimension: str, metric: str
    ) -> np.ndarray:
        """Gather a metric's value across all runs for one arch/dimension."""
        values = []
        for df in self._data[arch]:
            row = df[df["column"] == dimension]
            if row.empty:
                continue
            values.append(float(row.iloc[0][metric]))
        return np.array(values)

    @staticmethod
    def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Cohen's d for two samples using pooled std."""
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            return float("nan")
        var1, var2 = a.var(ddof=1), b.var(ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return float((a.mean() - b.mean()) / pooled_std)

    @staticmethod
    def _confidence_interval_95(arr: np.ndarray) -> tuple[float, float]:
        """Compute 95% CI for the mean using t-distribution."""
        n = len(arr)
        if n < 2:
            return (float("nan"), float("nan"))
        mean = arr.mean()
        se = arr.std(ddof=1) / np.sqrt(n)
        t_crit = sp_stats.t.ppf(0.975, df=n - 1)
        margin = t_crit * se
        return (float(mean - margin), float(mean + margin))

    # ------------------------------------------------------------------
    # Public summary methods
    # ------------------------------------------------------------------

    def descriptive_summary(self) -> pd.DataFrame:
        """Descriptive stats per architecture/dimension/metric.

        Columns: architecture, dimension, metric, n_runs, mean, std,
        cv_pct, min, max, range, ci_lower, ci_upper
        """
        rows = []
        for arch in self._arch_names:
            for dim in self.dimensions:
                for metric in self._metrics:
                    vals = self._collect_metric_values(arch, dim, metric)
                    if len(vals) == 0:
                        continue
                    mu = float(vals.mean())
                    sigma = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                    cv = (sigma / mu * 100) if mu != 0 else float("nan")
                    ci_lo, ci_hi = self._confidence_interval_95(vals)
                    rows.append({
                        "architecture": arch,
                        "dimension": dim,
                        "metric": metric,
                        "n_runs": len(vals),
                        "mean": round(mu, 4),
                        "std": round(sigma, 4),
                        "cv_pct": round(cv, 2),
                        "min": round(float(vals.min()), 4),
                        "max": round(float(vals.max()), 4),
                        "range": round(float(vals.max() - vals.min()), 4),
                        "ci_lower": round(ci_lo, 4),
                        "ci_upper": round(ci_hi, 4),
                    })
        return pd.DataFrame(rows)

    def paired_ttest_summary(self) -> pd.DataFrame:
        """Paired t-tests between all architecture pairs.

        Columns: dimension, metric, arch_a, arch_b, mean_a, mean_b,
        t_statistic, p_value, significant
        """
        rows = []
        pairs = list(combinations(self._arch_names, 2))
        for dim in self.dimensions:
            for metric in self._metrics:
                for arch_a, arch_b in pairs:
                    vals_a = self._collect_metric_values(arch_a, dim, metric)
                    vals_b = self._collect_metric_values(arch_b, dim, metric)
                    n = min(len(vals_a), len(vals_b))
                    if n < 2:
                        t_stat, p_val = float("nan"), float("nan")
                    else:
                        t_stat, p_val = sp_stats.ttest_rel(
                            vals_a[:n], vals_b[:n]
                        )
                    rows.append({
                        "dimension": dim,
                        "metric": metric,
                        "arch_a": arch_a,
                        "arch_b": arch_b,
                        "mean_a": round(float(vals_a.mean()), 4),
                        "mean_b": round(float(vals_b.mean()), 4),
                        "t_statistic": round(float(t_stat), 4),
                        "p_value": round(float(p_val), 6),
                        "significant": p_val < 0.05 if not np.isnan(p_val) else False,
                    })
        return pd.DataFrame(rows)

    def effect_size_summary(self) -> pd.DataFrame:
        """Cohen's d effect size between all architecture pairs.

        Columns: dimension, metric, arch_a, arch_b, cohens_d, magnitude
        """
        rows = []
        pairs = list(combinations(self._arch_names, 2))
        for dim in self.dimensions:
            for metric in self._metrics:
                for arch_a, arch_b in pairs:
                    vals_a = self._collect_metric_values(arch_a, dim, metric)
                    vals_b = self._collect_metric_values(arch_b, dim, metric)
                    d = self._cohens_d(vals_a, vals_b)
                    abs_d = abs(d) if not np.isnan(d) else float("nan")
                    if np.isnan(abs_d):
                        mag = "n/a"
                    elif abs_d < 0.2:
                        mag = "negligible"
                    elif abs_d < 0.5:
                        mag = "small"
                    elif abs_d < 0.8:
                        mag = "medium"
                    else:
                        mag = "large"
                    rows.append({
                        "dimension": dim,
                        "metric": metric,
                        "arch_a": arch_a,
                        "arch_b": arch_b,
                        "cohens_d": round(float(d), 4),
                        "magnitude": mag,
                    })
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        """Return the descriptive stats (consistent with evaluator API)."""
        return self.descriptive_summary()
