"""
transformations.py
-------------------
Feature transformation techniques for AutoEDA Pro: Log, Square Root,
Box-Cox, and Yeo-Johnson. Includes before/after distribution plots to
visualize the effect of each transformation on skewed numeric columns.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PowerTransformer

logger = logging.getLogger("autoeda.transformations")

METHODS = ["log", "sqrt", "boxcox", "yeojohnson"]


class FeatureTransformer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ------------------------------------------------------------------
    def transform_column(self, column: str, method: str = "log") -> pd.Series:
        series = self.df[column].astype(float)

        if method == "log":
            shift = abs(min(series.min(), 0)) + 1
            return np.log(series + shift)

        if method == "sqrt":
            shift = abs(min(series.min(), 0))
            return np.sqrt(series + shift)

        if method == "boxcox":
            shift = abs(min(series.min(), 0)) + 1e-6
            transformed, _ = stats.boxcox(series + shift)
            return pd.Series(transformed, index=series.index)

        if method == "yeojohnson":
            pt = PowerTransformer(method="yeo-johnson")
            transformed = pt.fit_transform(series.values.reshape(-1, 1)).ravel()
            return pd.Series(transformed, index=series.index)

        raise ValueError(f"Unknown transformation '{method}'. Choose from {METHODS}.")

    # ------------------------------------------------------------------
    def transform(self, columns: list, method: str = "yeojohnson") -> pd.DataFrame:
        df = self.df.copy()
        for col in columns:
            try:
                df[col] = self.transform_column(col, method)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Transformation '%s' failed on column '%s': %s", method, col, exc)
        return df

    # ------------------------------------------------------------------
    def recommend_columns(self, skew_threshold: float = 1.0) -> list:
        """Return numeric columns whose absolute skewness exceeds the threshold."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        return [c for c in numeric_df.columns if abs(numeric_df[c].dropna().skew()) > skew_threshold]

    # ------------------------------------------------------------------
    def plot_before_after(self, column: str, method: str = "yeojohnson",
                           output_dir: str = "charts",
                           filename: Optional[str] = None) -> Optional[str]:
        """Save a side-by-side before/after distribution plot for one column."""
        try:
            after = self.transform_column(column, method)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not plot transformation for '%s': %s", column, exc)
            return None

        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        sns.histplot(self.df[column].dropna(), kde=True, ax=axes[0], color="#DD8452")
        axes[0].set_title(f"{column} - Before ({method})")
        sns.histplot(after.dropna(), kde=True, ax=axes[1], color="#4C72B0")
        axes[1].set_title(f"{column} - After ({method})")
        fig.tight_layout()

        filename = filename or f"transform_{column}_{method}.png"
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
