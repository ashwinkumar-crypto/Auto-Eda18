"""
eda.py
------
Exploratory Data Analysis engine for AutoEDA Pro.

Produces:
  - Dataset overview (shape, dtypes, memory, missing, duplicates)
  - Statistical summary (mean, median, mode, min, max, variance, std,
    quartiles, skewness, kurtosis)
  - Missing value analysis
  - Duplicate analysis
  - Correlation analysis (Pearson, Spearman, Kendall)
"""

import numpy as np
import pandas as pd

from utils import get_memory_usage, is_numeric_column


class EDAAnalyzer:
    def __init__(self, df):
        self.df = df

    # ------------------------------------------------------------------
    def dataset_overview(self):
        df = self.df
        return {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "memory_usage": get_memory_usage(df),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values_total": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        }

    # ------------------------------------------------------------------
    def dataset_information(self):
        df = self.df
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "non_null_counts": df.notnull().sum().to_dict(),
        }

    # ------------------------------------------------------------------
    def statistical_summary(self):
        df = self.df
        numeric_df = df.select_dtypes(include=[np.number])
        summary = {}

        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if series.empty:
                continue
            mode_val = series.mode()
            summary[col] = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "mode": float(mode_val.iloc[0]) if not mode_val.empty else None,
                "min": float(series.min()),
                "max": float(series.max()),
                "variance": float(series.var()),
                "std": float(series.std()),
                "q1": float(series.quantile(0.25)),
                "q3": float(series.quantile(0.75)),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurt()),
            }
        return summary

    # ------------------------------------------------------------------
    def missing_value_analysis(self):
        df = self.df
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / len(df) * 100).round(2)
        table = pd.DataFrame({
            "missing_count": missing_count,
            "missing_percent": missing_pct,
        })
        table = table[table["missing_count"] > 0].sort_values(
            "missing_count", ascending=False
        )
        return table

    # ------------------------------------------------------------------
    def duplicate_analysis(self):
        df = self.df
        dup_count = int(df.duplicated().sum())
        dup_pct = round((dup_count / len(df) * 100), 2) if len(df) else 0
        return {"duplicate_count": dup_count, "duplicate_percent": dup_pct}

    # ------------------------------------------------------------------
    def correlation_analysis(self, include_kendall=False):
        df = self.df
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            return {
                "pearson": None,
                "spearman": None,
                "kendall": None,
                "strongest_positive": None,
                "strongest_negative": None,
            }

        pearson = numeric_df.corr(method="pearson")
        spearman = numeric_df.corr(method="spearman")
        kendall = numeric_df.corr(method="kendall") if include_kendall else None

        strongest_pos, strongest_neg = self._strongest_correlations(pearson)

        return {
            "pearson": pearson,
            "spearman": spearman,
            "kendall": kendall,
            "strongest_positive": strongest_pos,
            "strongest_negative": strongest_neg,
        }

    @staticmethod
    def _strongest_correlations(corr_matrix):
        """Find the strongest positive and negative pairwise correlations,
        ignoring the diagonal and duplicate (i, j)/(j, i) pairs."""
        corr = corr_matrix.copy()
        corr_arr = corr.to_numpy(copy=True)
        np.fill_diagonal(corr_arr, np.nan)
        corr = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)

        pairs = corr.unstack().dropna()
        # Remove duplicate mirrored pairs (A,B) vs (B,A)
        pairs = pairs[~pairs.index.duplicated()]
        seen = set()
        unique_pairs = {}
        for (a, b), val in pairs.items():
            key = frozenset([a, b])
            if key in seen or a == b:
                continue
            seen.add(key)
            unique_pairs[(a, b)] = val

        if not unique_pairs:
            return None, None

        strongest_pos = max(unique_pairs.items(), key=lambda x: x[1])
        strongest_neg = min(unique_pairs.items(), key=lambda x: x[1])
        return strongest_pos, strongest_neg
