"""
feature_engineering.py
-----------------------
Feature engineering utilities for AutoEDA Pro: datetime feature
extraction, text length features, polynomial features, and pairwise
interaction features. Designed to be applied selectively (not all
transformations make sense for every dataset), so each method is
opt-in and returns a new DataFrame.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from utils import is_datetime_like

logger = logging.getLogger("autoeda.feature_engineering")


class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.log: dict = {"datetime_columns_expanded": [], "text_length_columns": [],
                           "polynomial_source_columns": [], "interaction_pairs": []}

    # ------------------------------------------------------------------
    def extract_datetime_features(self, columns: Optional[list] = None) -> pd.DataFrame:
        """Expand datetime-like columns into year/month/day/dayofweek/is_weekend."""
        df = self.df.copy()
        candidate_cols = columns or [c for c in df.columns if is_datetime_like(df[c], name_hint=c)]

        for col in candidate_cols:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                continue
            if parsed.isna().all():
                continue
            df[f"{col}_year"] = parsed.dt.year
            df[f"{col}_month"] = parsed.dt.month
            df[f"{col}_day"] = parsed.dt.day
            df[f"{col}_dayofweek"] = parsed.dt.dayofweek
            df[f"{col}_is_weekend"] = (parsed.dt.dayofweek >= 5).astype(int)
            self.log["datetime_columns_expanded"].append(col)

        self.df = df
        return df

    # ------------------------------------------------------------------
    def add_text_length_features(self, columns: Optional[list] = None,
                                  min_avg_length: float = 8.0) -> pd.DataFrame:
        """Add a `<col>_length` feature for free-text-like string columns."""
        df = self.df.copy()
        if columns is None:
            candidates = df.select_dtypes(include=["object"]).columns.tolist()
            candidates = [c for c in candidates if not is_datetime_like(df[c], name_hint=c)]
            columns = [
                c for c in candidates
                if df[c].dropna().astype(str).str.len().mean() >= min_avg_length
            ]

        for col in columns:
            df[f"{col}_length"] = df[col].astype(str).str.len()
            self.log["text_length_columns"].append(col)

        self.df = df
        return df

    # ------------------------------------------------------------------
    def add_polynomial_features(self, columns: Optional[list] = None, degree: int = 2,
                                 max_columns: int = 5) -> pd.DataFrame:
        """Add polynomial (and interaction) terms for the given numeric columns."""
        df = self.df.copy()
        numeric_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()[:max_columns]
        numeric_cols = [c for c in numeric_cols if c in df.columns][:max_columns]
        if len(numeric_cols) < 1:
            return df

        data = df[numeric_cols].fillna(df[numeric_cols].median())
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        transformed = poly.fit_transform(data)
        feature_names = poly.get_feature_names_out(numeric_cols)

        new_cols = {}
        for name, values in zip(feature_names, transformed.T):
            if name in numeric_cols:
                continue  # skip the original (degree-1) columns, already present
            safe_name = name.replace(" ", "_").replace("^", "_pow")
            new_cols[f"poly_{safe_name}"] = values

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        self.log["polynomial_source_columns"] = numeric_cols
        self.df = df
        return df

    # ------------------------------------------------------------------
    def add_interaction_features(self, columns: Optional[list] = None,
                                  max_pairs: int = 10) -> pd.DataFrame:
        """Add pairwise product interaction features between numeric columns."""
        df = self.df.copy()
        numeric_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        pairs = list(combinations(numeric_cols, 2))[:max_pairs]

        for a, b in pairs:
            df[f"{a}_x_{b}"] = df[a] * df[b]
            self.log["interaction_pairs"].append((a, b))

        self.df = df
        return df

    # ------------------------------------------------------------------
    def run_all(self, include_polynomial: bool = False, include_interactions: bool = False,
                polynomial_degree: int = 2, max_interaction_pairs: int = 6) -> pd.DataFrame:
        """Run the safe, generally-applicable feature engineering steps.
        Polynomial and interaction features are opt-in since they can
        rapidly expand dimensionality on wide datasets."""
        self.extract_datetime_features()
        self.add_text_length_features()
        if include_polynomial:
            self.add_polynomial_features(degree=polynomial_degree)
        if include_interactions:
            self.add_interaction_features(max_pairs=max_interaction_pairs)
        return self.df

    def get_log(self) -> dict:
        return self.log
