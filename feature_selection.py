"""
feature_selection.py
---------------------
Feature selection techniques for AutoEDA Pro: Correlation Threshold,
Variance Threshold, SelectKBest, Recursive Feature Elimination (RFE),
Mutual Information, Chi-Square, Random Forest importance, and Lasso.
Produces a ranked feature-importance report combining available methods.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, f_regression,
    mutual_info_classif, mutual_info_regression, chi2, RFE,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

logger = logging.getLogger("autoeda.feature_selection")


class FeatureSelector:
    def __init__(self, df: pd.DataFrame, target_column: str):
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")
        self.df = df
        self.target_column = target_column

        numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[target_column], errors="ignore")
        self.feature_names = numeric_df.columns.tolist()
        self.X = numeric_df.fillna(numeric_df.median())

        y_raw = df[target_column]
        self.is_classification = (not pd.api.types.is_numeric_dtype(y_raw)) or y_raw.nunique() <= 20
        if not pd.api.types.is_numeric_dtype(y_raw):
            self.y = pd.Series(LabelEncoder().fit_transform(y_raw.astype(str)), index=df.index)
        else:
            self.y = y_raw.fillna(y_raw.median())

    # ------------------------------------------------------------------
    def correlation_threshold(self, threshold: float = 0.1) -> pd.Series:
        if self.X.empty:
            return pd.Series(dtype=float)
        corr = self.X.assign(**{self.target_column: self.y}).corr()[self.target_column].drop(self.target_column)
        return corr.abs().sort_values(ascending=False)[lambda s: s >= threshold]

    def variance_threshold(self, threshold: float = 0.0) -> list:
        if self.X.empty:
            return []
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(self.X)
        return [f for f, keep in zip(self.feature_names, selector.get_support()) if keep]

    def select_k_best(self, k: int = 10) -> pd.Series:
        if self.X.empty:
            return pd.Series(dtype=float)
        k = min(k, self.X.shape[1])
        score_func = f_classif if self.is_classification else f_regression
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(self.X, self.y)
        scores = pd.Series(selector.scores_, index=self.feature_names).fillna(0)
        return scores.sort_values(ascending=False).head(k)

    def mutual_information(self, k: int = 10) -> pd.Series:
        if self.X.empty:
            return pd.Series(dtype=float)
        func = mutual_info_classif if self.is_classification else mutual_info_regression
        scores = func(self.X, self.y, random_state=42)
        return pd.Series(scores, index=self.feature_names).sort_values(ascending=False).head(k)

    def chi_square(self, k: int = 10) -> Optional[pd.Series]:
        """Chi-square requires non-negative features; only meaningful for classification."""
        if not self.is_classification or self.X.empty:
            return None
        X_nonneg = MinMaxScaler().fit_transform(self.X)
        k = min(k, self.X.shape[1])
        selector = SelectKBest(score_func=chi2, k=k)
        selector.fit(X_nonneg, self.y)
        scores = pd.Series(selector.scores_, index=self.feature_names).fillna(0)
        return scores.sort_values(ascending=False).head(k)

    def rfe_ranking(self, n_features: int = 10) -> list:
        if self.X.empty:
            return []
        n_features = min(n_features, self.X.shape[1])
        estimator = (LogisticRegression(max_iter=1000) if self.is_classification
                     else Lasso(alpha=0.01, max_iter=5000))
        selector = RFE(estimator, n_features_to_select=n_features)
        selector.fit(self.X, self.y)
        return [f for f, keep in zip(self.feature_names, selector.support_) if keep]

    def random_forest_importance(self) -> pd.Series:
        if self.X.empty:
            return pd.Series(dtype=float)
        model = (RandomForestClassifier(n_estimators=200, random_state=42) if self.is_classification
                  else RandomForestRegressor(n_estimators=200, random_state=42))
        model.fit(self.X, self.y)
        return pd.Series(model.feature_importances_, index=self.feature_names).sort_values(ascending=False)

    def lasso_selection(self, alpha: float = 0.01) -> pd.Series:
        if self.X.empty:
            return pd.Series(dtype=float)
        model = Lasso(alpha=alpha, max_iter=5000)
        model.fit(self.X, self.y)
        coefs = pd.Series(np.abs(model.coef_), index=self.feature_names)
        return coefs.sort_values(ascending=False)

    # ------------------------------------------------------------------
    def ranked_report(self, top_k: int = 15) -> pd.DataFrame:
        """
        Combine Random Forest importance, mutual information, and
        |correlation| into a single ranked feature-importance report,
        normalized to [0, 1] and averaged for a composite rank.
        """
        if self.X.empty:
            return pd.DataFrame(columns=["feature", "rf_importance", "mutual_info", "abs_correlation", "composite_score"])

        rf = self.random_forest_importance()
        mi = self.mutual_information(k=len(self.feature_names))
        corr = self.X.assign(**{self.target_column: self.y}).corr()[self.target_column].drop(self.target_column).abs()

        def _norm(s):
            s = s.reindex(self.feature_names).fillna(0)
            rng = s.max() - s.min()
            return (s - s.min()) / rng if rng > 0 else s * 0

        rf_n, mi_n, corr_n = _norm(rf), _norm(mi), _norm(corr)
        composite = (rf_n + mi_n + corr_n) / 3

        report = pd.DataFrame({
            "feature": self.feature_names,
            "rf_importance": rf.reindex(self.feature_names).fillna(0).values,
            "mutual_info": mi.reindex(self.feature_names).fillna(0).values,
            "abs_correlation": corr.reindex(self.feature_names).fillna(0).values,
            "composite_score": composite.values,
        }).sort_values("composite_score", ascending=False).reset_index(drop=True)

        return report.head(top_k)
