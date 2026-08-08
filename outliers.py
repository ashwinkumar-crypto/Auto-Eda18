"""
outliers.py
-----------
Multi-method outlier detection for AutoEDA Pro: IQR, Z-Score, Isolation
Forest, Local Outlier Factor, and DBSCAN. Each method returns a boolean
mask (True = keep, False = outlier) plus a summary dict so the caller
can report counts/percentages and produce before/after visualizations.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("autoeda.outliers")

METHODS = ["iqr", "zscore", "isolation_forest", "lof", "dbscan"]


class OutlierDetector:
    def __init__(self, df: pd.DataFrame, columns: Optional[list] = None):
        self.df = df
        self.columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()

    # ------------------------------------------------------------------
    def _numeric_matrix(self):
        data = self.df[self.columns].copy()
        data = data.fillna(data.median(numeric_only=True))
        return data

    def detect(self, method: str = "iqr", **kwargs) -> dict:
        """
        Returns {"mask": pd.Series[bool] (True=keep), "n_outliers": int,
        "percent": float, "method": str}.
        """
        if not self.columns:
            mask = pd.Series(True, index=self.df.index)
            return {"mask": mask, "n_outliers": 0, "percent": 0.0, "method": method}

        if method == "iqr":
            mask = self._iqr(kwargs.get("factor", 1.5))
        elif method == "zscore":
            mask = self._zscore(kwargs.get("threshold", 3.0))
        elif method == "isolation_forest":
            mask = self._isolation_forest(kwargs.get("contamination", 0.05))
        elif method == "lof":
            mask = self._lof(kwargs.get("contamination", 0.05), kwargs.get("n_neighbors", 20))
        elif method == "dbscan":
            mask = self._dbscan(kwargs.get("eps", 1.5), kwargs.get("min_samples", 5))
        else:
            raise ValueError(f"Unknown outlier method '{method}'. Choose from {METHODS}.")

        n_outliers = int((~mask).sum())
        percent = round(n_outliers / len(self.df) * 100, 2) if len(self.df) else 0.0
        return {"mask": mask, "n_outliers": n_outliers, "percent": percent, "method": method}

    # ------------------------------------------------------------------
    def _iqr(self, factor: float) -> pd.Series:
        df = self.df
        mask = pd.Series(True, index=df.index)
        for col in self.columns:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - factor * iqr, q3 + factor * iqr
            mask &= df[col].between(lower, upper) | df[col].isna()
        return mask

    def _zscore(self, threshold: float) -> pd.Series:
        df = self.df
        mask = pd.Series(True, index=df.index)
        for col in self.columns:
            series = df[col]
            std = series.std()
            if not std or np.isnan(std):
                continue
            z = (series - series.mean()) / std
            mask &= (z.abs() <= threshold) | series.isna()
        return mask

    def _isolation_forest(self, contamination: float) -> pd.Series:
        data = self._numeric_matrix()
        model = IsolationForest(contamination=contamination, random_state=42)
        preds = model.fit_predict(data)  # -1 = outlier, 1 = inlier
        return pd.Series(preds == 1, index=self.df.index)

    def _lof(self, contamination: float, n_neighbors: int) -> pd.Series:
        data = self._numeric_matrix()
        n_neighbors = min(n_neighbors, max(len(data) - 1, 1))
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        preds = model.fit_predict(data)
        return pd.Series(preds == 1, index=self.df.index)

    def _dbscan(self, eps: float, min_samples: int) -> pd.Series:
        data = self._numeric_matrix()
        scaled = StandardScaler().fit_transform(data)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(scaled)  # -1 = noise/outlier
        return pd.Series(labels != -1, index=self.df.index)

    # ------------------------------------------------------------------
    def compare_methods(self, methods: Optional[list] = None) -> pd.DataFrame:
        """Run several methods and summarize outlier counts for comparison."""
        methods = methods or METHODS
        rows = []
        for m in methods:
            try:
                result = self.detect(m)
                rows.append({
                    "method": m,
                    "n_outliers": result["n_outliers"],
                    "percent_removed": result["percent"],
                })
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Outlier method '%s' failed: %s", m, exc)
                rows.append({"method": m, "n_outliers": None, "percent_removed": None})
        return pd.DataFrame(rows)
