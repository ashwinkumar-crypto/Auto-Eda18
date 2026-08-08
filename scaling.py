"""
scaling.py
----------
Feature scaling strategies for AutoEDA Pro: Min-Max, Standard, Robust,
MaxAbs, and Normalizer, with an automatic recommendation heuristic
based on outlier prevalence and distribution shape.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, Normalizer,
)

logger = logging.getLogger("autoeda.scaling")

METHODS = ["minmax", "standard", "robust", "maxabs", "normalizer"]

_SCALER_CLASSES = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
    "robust": RobustScaler,
    "maxabs": MaxAbsScaler,
    "normalizer": Normalizer,
}


class FeatureScaler:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.log: dict = {}

    # ------------------------------------------------------------------
    def scale(self, method: str = "minmax", columns: Optional[list] = None) -> pd.DataFrame:
        df = self.df.copy()
        numeric_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df
        if method not in _SCALER_CLASSES:
            raise ValueError(f"Unknown scaling method '{method}'. Choose from {METHODS}.")

        scaler = _SCALER_CLASSES[method]()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.log = {"method": method, "columns_scaled": numeric_cols}
        return df

    # ------------------------------------------------------------------
    def recommend_method(self, outlier_fraction_threshold: float = 0.05) -> str:
        """
        Heuristic:
          - No numeric columns -> 'none'
          - Significant outliers present (by IQR rule) -> robust (resistant to outliers)
          - Distributions roughly Gaussian (|skew| < 0.5 on average) -> standard
          - Otherwise -> minmax (bounded [0, 1], safe default for mixed distributions)
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return "none"

        outlier_fractions = []
        skews = []
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                outlier_fractions.append(0.0)
            else:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_fractions.append(((series < lower) | (series > upper)).mean())
            skews.append(abs(series.skew()))

        avg_outlier_frac = np.mean(outlier_fractions) if outlier_fractions else 0.0
        avg_skew = np.mean(skews) if skews else 0.0

        if avg_outlier_frac > outlier_fraction_threshold:
            return "robust"
        if avg_skew < 0.5:
            return "standard"
        return "minmax"
