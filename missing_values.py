"""
missing_values.py
------------------
Advanced missing-value handling for AutoEDA Pro.

Supports Drop Rows, Drop Columns, Mean/Median/Mode/Constant imputation,
Forward Fill, Backward Fill, KNN Imputer, and Iterative Imputer (MICE),
plus a comparison table across techniques with an automatic
recommendation. Complements (does not replace) the simple median/mode
imputation already used by DataPreprocessor.handle_missing_values().
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer

from utils import is_numeric_column

logger = logging.getLogger("autoeda.missing_values")

TECHNIQUES = [
    "drop_rows", "drop_columns", "mean", "median", "mode", "constant",
    "ffill", "bfill", "knn", "mice",
]


class MissingValueHandler:
    """Applies a chosen missing-value technique and can compare several."""

    def __init__(self, df: pd.DataFrame):
        self.original_df = df.copy()

    # ------------------------------------------------------------------
    def apply(self, technique: str, constant_value=0, knn_neighbors: int = 5,
              drop_column_threshold: float = 0.5) -> pd.DataFrame:
        """Apply a single missing-value technique and return the result."""
        df = self.original_df.copy()

        if technique == "drop_rows":
            return df.dropna().reset_index(drop=True)

        if technique == "drop_columns":
            missing_frac = df.isnull().mean()
            cols_to_drop = missing_frac[missing_frac > drop_column_threshold].index.tolist()
            return df.drop(columns=cols_to_drop)

        if technique == "ffill":
            return df.ffill().bfill()  # bfill as a safety net for leading NaNs

        if technique == "bfill":
            return df.bfill().ffill()

        numeric_cols = [c for c in df.columns if is_numeric_column(df[c])]
        other_cols = [c for c in df.columns if c not in numeric_cols]

        if technique == "mean":
            for c in numeric_cols:
                df[c] = df[c].fillna(df[c].mean())
            for c in other_cols:
                df[c] = df[c].fillna(self._mode_or_default(df[c]))
            return df

        if technique == "median":
            for c in numeric_cols:
                df[c] = df[c].fillna(df[c].median())
            for c in other_cols:
                df[c] = df[c].fillna(self._mode_or_default(df[c]))
            return df

        if technique == "mode":
            for c in df.columns:
                df[c] = df[c].fillna(self._mode_or_default(df[c]))
            return df

        if technique == "constant":
            for c in numeric_cols:
                df[c] = df[c].fillna(constant_value)
            for c in other_cols:
                df[c] = df[c].fillna(str(constant_value))
            return df

        if technique == "knn":
            if not numeric_cols:
                logger.warning("KNN imputation skipped: no numeric columns.")
                return df
            imputer = KNNImputer(n_neighbors=knn_neighbors)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            for c in other_cols:
                df[c] = df[c].fillna(self._mode_or_default(df[c]))
            return df

        if technique == "mice":
            if not numeric_cols:
                logger.warning("MICE imputation skipped: no numeric columns.")
                return df
            imputer = IterativeImputer(random_state=42, max_iter=10)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            for c in other_cols:
                df[c] = df[c].fillna(self._mode_or_default(df[c]))
            return df

        raise ValueError(f"Unknown missing-value technique: '{technique}'. Choose from {TECHNIQUES}.")

    @staticmethod
    def _mode_or_default(series: pd.Series, default="Unknown"):
        mode_series = series.mode(dropna=True)
        return mode_series.iloc[0] if not mode_series.empty else default

    # ------------------------------------------------------------------
    def compare_techniques(self, techniques: Optional[list] = None) -> pd.DataFrame:
        """
        Run several techniques and build a comparison table showing, per
        technique: which columns it touched, how many missing values it
        filled, resulting row/column counts, and whether it is recommended.
        """
        techniques = techniques or [t for t in TECHNIQUES if t not in ("drop_rows", "drop_columns")]
        original_missing = int(self.original_df.isnull().sum().sum())
        total_cols_with_na = self.original_df.columns[self.original_df.isnull().any()].tolist()

        rows = []
        for tech in techniques:
            try:
                result_df = self.apply(tech)
                remaining_missing = int(result_df.isnull().sum().sum())
                filled = original_missing - remaining_missing if tech not in ("drop_rows",) else original_missing
                rows.append({
                    "technique": tech,
                    "columns_applied": ", ".join(total_cols_with_na) if total_cols_with_na else "-",
                    "missing_filled": filled,
                    "rows_after": result_df.shape[0],
                    "columns_after": result_df.shape[1],
                    "remaining_missing": remaining_missing,
                })
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Technique '%s' failed during comparison: %s", tech, exc)
                rows.append({
                    "technique": tech, "columns_applied": "-", "missing_filled": 0,
                    "rows_after": None, "columns_after": None, "remaining_missing": None,
                })

        comparison = pd.DataFrame(rows)
        recommended = self.recommend_technique()
        comparison["recommended"] = comparison["technique"] == recommended
        return comparison

    # ------------------------------------------------------------------
    def recommend_technique(self) -> str:
        """
        Heuristic recommendation:
          - No missing values -> 'none'
          - < 3% missing overall and dataset is large -> drop_rows
          - Any column > 50% missing -> drop_columns (in combination with imputation)
          - Numeric-heavy dataset with moderate missingness -> mice (best statistical fidelity)
          - Small dataset or few numeric columns -> median (safe, fast, robust to outliers)
        """
        df = self.original_df
        total_missing = df.isnull().sum().sum()
        if total_missing == 0:
            return "none"

        missing_frac_overall = total_missing / df.size
        col_missing_frac = df.isnull().mean()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if (col_missing_frac > 0.5).any():
            return "drop_columns"
        if missing_frac_overall < 0.03 and len(df) > 1000:
            return "drop_rows"
        if len(numeric_cols) >= 3 and 0.03 <= missing_frac_overall < 0.3:
            return "mice"
        return "median"
