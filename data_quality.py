"""
data_quality.py
----------------
Computes a Data Quality Report and an overall Data Quality Score (0-100)
for a dataset, covering missingness, duplication, constant/near-constant
columns, high-cardinality columns, zero-variance columns, highly
correlated feature pairs, memory usage, and (if a target column is
supplied) class distribution / imbalance.

Used by both the CLI pipeline (main.py) and the Streamlit app (app.py).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from utils import get_memory_usage, is_datetime_like

logger = logging.getLogger("autoeda.data_quality")


class DataQualityReport:
    """Builds a structured data-quality report and score for a DataFrame."""

    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None,
                 high_cardinality_threshold: int = 50,
                 correlation_threshold: float = 0.9):
        self.df = df
        self.target_column = target_column
        self.high_cardinality_threshold = high_cardinality_threshold
        self.correlation_threshold = correlation_threshold
        self.report: dict = {}

    # ------------------------------------------------------------------
    def _missing_summary(self) -> dict:
        df = self.df
        total_cells = df.shape[0] * df.shape[1] if df.size else 1
        missing_total = int(df.isnull().sum().sum())
        pct = round(missing_total / total_cells * 100, 2) if total_cells else 0.0
        by_col = (df.isnull().mean() * 100).round(2)
        by_col = by_col[by_col > 0].sort_values(ascending=False).to_dict()
        return {"total_missing": missing_total, "missing_percent": pct, "by_column": by_col}

    def _duplicate_summary(self) -> dict:
        df = self.df
        dup_count = int(df.duplicated().sum())
        pct = round(dup_count / len(df) * 100, 2) if len(df) else 0.0
        return {"duplicate_count": dup_count, "duplicate_percent": pct}

    def _dtype_summary(self) -> dict:
        return self.df.dtypes.astype(str).value_counts().to_dict()

    def _constant_columns(self) -> list:
        df = self.df
        return [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

    def _high_cardinality_columns(self) -> list:
        df = self.df
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        cat_cols = [c for c in cat_cols if not is_datetime_like(df[c], name_hint=c)]
        return [c for c in cat_cols if df[c].nunique(dropna=True) > self.high_cardinality_threshold]

    def _zero_variance_columns(self) -> list:
        df = self.df
        numeric_df = df.select_dtypes(include=[np.number])
        return [c for c in numeric_df.columns if numeric_df[c].std(skipna=True) == 0]

    def _highly_correlated_pairs(self) -> list:
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return []
        corr = numeric_df.corr().abs()
        corr_arr = corr.to_numpy(copy=True)
        np.fill_diagonal(corr_arr, 0)
        corr = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)
        pairs = []
        seen = set()
        for col in corr.columns:
            for row in corr.index:
                if row == col:
                    continue
                key = frozenset([row, col])
                if key in seen:
                    continue
                seen.add(key)
                val = corr.loc[row, col]
                if val >= self.correlation_threshold:
                    pairs.append((row, col, round(float(val), 3)))
        return sorted(pairs, key=lambda x: x[2], reverse=True)

    def _class_distribution(self) -> Optional[dict]:
        if not self.target_column or self.target_column not in self.df.columns:
            return None
        counts = self.df[self.target_column].value_counts(dropna=False)
        pct = (counts / counts.sum() * 100).round(2)
        imbalance_ratio = round(float(counts.max() / max(counts.min(), 1)), 2)
        return {
            "counts": counts.to_dict(),
            "percent": pct.to_dict(),
            "imbalance_ratio": imbalance_ratio,
            "is_imbalanced": imbalance_ratio >= 3.0,
        }

    # ------------------------------------------------------------------
    def _compute_score(self, missing, duplicate, constant_cols, high_card_cols,
                        zero_var_cols, corr_pairs) -> int:
        """Weighted deduction-based quality score, clamped to [0, 100]."""
        score = 100.0
        score -= min(missing["missing_percent"] * 1.2, 35)
        score -= min(duplicate["duplicate_percent"] * 0.8, 20)
        n_cols = max(self.df.shape[1], 1)
        score -= min(len(constant_cols) / n_cols * 100 * 0.5, 15)
        score -= min(len(zero_var_cols) / n_cols * 100 * 0.5, 10)
        score -= min(len(high_card_cols) * 2, 10)
        score -= min(len(corr_pairs) * 1.5, 10)
        return int(round(max(0, min(100, score))))

    @staticmethod
    def _grade(score: int) -> str:
        if score >= 90:
            return "Excellent"
        if score >= 75:
            return "Good"
        if score >= 60:
            return "Fair"
        if score >= 40:
            return "Poor"
        return "Critical"

    # ------------------------------------------------------------------
    def generate(self) -> dict:
        """Compute and return the full data quality report dict."""
        logger.info("Generating data quality report ...")
        missing = self._missing_summary()
        duplicate = self._duplicate_summary()
        constant_cols = self._constant_columns()
        high_card_cols = self._high_cardinality_columns()
        zero_var_cols = self._zero_variance_columns()
        corr_pairs = self._highly_correlated_pairs()
        class_dist = self._class_distribution()

        score = self._compute_score(missing, duplicate, constant_cols,
                                     high_card_cols, zero_var_cols, corr_pairs)

        self.report = {
            "shape": self.df.shape,
            "memory_usage": get_memory_usage(self.df),
            "missing": missing,
            "duplicate": duplicate,
            "dtype_summary": self._dtype_summary(),
            "constant_columns": constant_cols,
            "high_cardinality_columns": high_card_cols,
            "zero_variance_columns": zero_var_cols,
            "highly_correlated_pairs": corr_pairs,
            "class_distribution": class_dist,
            "quality_score": score,
            "quality_grade": self._grade(score),
        }
        logger.info("Data quality score: %s (%s)", score, self._grade(score))
        return self.report

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten the report into a two-column summary DataFrame for display."""
        if not self.report:
            self.generate()
        r = self.report
        rows = [
            ("Rows", r["shape"][0]),
            ("Columns", r["shape"][1]),
            ("Memory Usage", r["memory_usage"]),
            ("Missing Values (%)", r["missing"]["missing_percent"]),
            ("Duplicate Rows (%)", r["duplicate"]["duplicate_percent"]),
            ("Constant Columns", len(r["constant_columns"])),
            ("Zero-Variance Columns", len(r["zero_variance_columns"])),
            ("High-Cardinality Columns", len(r["high_cardinality_columns"])),
            ("Highly Correlated Pairs (>= threshold)", len(r["highly_correlated_pairs"])),
            ("Data Quality Score", f"{r['quality_score']} / 100"),
            ("Data Quality Grade", r["quality_grade"]),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Value"])
