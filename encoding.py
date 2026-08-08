"""
encoding.py
-----------
Categorical encoding strategies for AutoEDA Pro: Label, One-Hot,
Ordinal, Frequency, and Target Encoding, with an automatic
recommendation heuristic based on cardinality and target availability.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from utils import is_categorical_column, is_datetime_like

logger = logging.getLogger("autoeda.encoding")

METHODS = ["label", "onehot", "ordinal", "frequency", "target"]


class CategoricalEncoder:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.log: dict = {}

    # ------------------------------------------------------------------
    def encode(self, method: str = "label", columns: Optional[list] = None,
               target_column: Optional[str] = None,
               low_cardinality_max: int = 10) -> pd.DataFrame:
        df = self.df.copy()
        cat_cols = columns or [c for c in df.columns if is_categorical_column(df[c])
                                and c != target_column and not is_datetime_like(df[c], name_hint=c)]
        if not cat_cols:
            return df

        if method == "label":
            for col in cat_cols:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        elif method == "ordinal":
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            df[cat_cols] = encoder.fit_transform(df[cat_cols].astype(str))

        elif method == "onehot":
            df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

        elif method == "frequency":
            for col in cat_cols:
                freq = df[col].value_counts(normalize=True)
                df[col] = df[col].map(freq).fillna(0.0)

        elif method == "target":
            if not target_column or target_column not in df.columns:
                raise ValueError("Target encoding requires a valid `target_column`.")
            target = df[target_column]
            if not pd.api.types.is_numeric_dtype(target):
                target = LabelEncoder().fit_transform(target.astype(str))
                target = pd.Series(target, index=df.index)
            global_mean = target.mean()
            for col in cat_cols:
                means = target.groupby(df[col]).mean()
                df[col] = df[col].map(means).fillna(global_mean)

        else:
            raise ValueError(f"Unknown encoding method '{method}'. Choose from {METHODS}.")

        self.log = {"method": method, "columns_encoded": cat_cols}
        return df

    # ------------------------------------------------------------------
    def recommend_method(self, target_column: Optional[str] = None,
                          low_cardinality_max: int = 10) -> str:
        """
        Heuristic:
          - No categorical columns -> 'none'
          - A usable target exists -> target encoding (best for tree/linear models)
          - All categorical columns are low-cardinality -> one-hot (safe, no ordinal bias)
          - Any high-cardinality column present -> frequency encoding (compact, avoids
            explosion of one-hot columns)
        """
        cat_cols = [c for c in self.df.columns if is_categorical_column(self.df[c])
                    and c != target_column and not is_datetime_like(self.df[c], name_hint=c)]
        if not cat_cols:
            return "none"
        if target_column and target_column in self.df.columns:
            return "target"
        cardinalities = [self.df[c].nunique(dropna=True) for c in cat_cols]
        if max(cardinalities) > low_cardinality_max:
            return "frequency"
        return "onehot"
