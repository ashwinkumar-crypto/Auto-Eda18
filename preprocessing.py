"""
preprocessing.py
----------------
Automated data preprocessing engine for AutoEDA Pro.

Handles:
  - Missing value imputation (median for numeric, mode for categorical)
  - Duplicate row removal
  - Outlier detection & removal using the IQR method
  - Categorical encoding (Label Encoding / optional One-Hot Encoding)
  - Feature scaling (Min-Max Normalization / Standardization)
  - General cleaning (constant columns, dtype fixes)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from utils import is_numeric_column, is_categorical_column


class DataPreprocessor:
    def __init__(self, df):
        self.original_df = df.copy()
        self.df = df.copy()
        self.log = {
            "missing_before": {},
            "missing_filled": {},
            "duplicates_removed": 0,
            "outliers_removed": 0,
            "constant_columns_dropped": [],
            "encoded_columns": [],
            "scaled_columns": [],
        }

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------
    def handle_missing_values(self):
        df = self.df
        self.log["missing_before"] = df.isnull().sum().to_dict()

        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue

            if is_numeric_column(df[col]):
                fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
                self.log["missing_filled"][col] = f"median ({fill_value:.4f})"
            else:
                mode_series = df[col].mode(dropna=True)
                fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                df[col] = df[col].fillna(fill_value)
                self.log["missing_filled"][col] = f"mode ({fill_value})"

        self.df = df
        return self

    # ------------------------------------------------------------------
    # Duplicates
    # ------------------------------------------------------------------
    def remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        self.log["duplicates_removed"] = before - len(self.df)
        return self

    # ------------------------------------------------------------------
    # Outliers (IQR method)
    # ------------------------------------------------------------------
    def remove_outliers(self, columns=None, factor=1.5):
        df = self.df
        numeric_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        before = len(df)
        mask = pd.Series(True, index=df.index)

        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            mask &= df[col].between(lower, upper)

        self.df = df[mask].reset_index(drop=True)
        self.log["outliers_removed"] = before - len(self.df)
        return self

    # ------------------------------------------------------------------
    # Cleaning: constant columns, dtype fixes
    # ------------------------------------------------------------------
    def clean_data(self):
        df = self.df
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if constant_cols:
            df = df.drop(columns=constant_cols)
            self.log["constant_columns_dropped"] = constant_cols
        self.df = df
        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode_categorical(self, method="label", columns=None):
        df = self.df
        cat_cols = columns or [c for c in df.columns if is_categorical_column(df[c])]

        if method == "onehot":
            if cat_cols:
                df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
                self.log["encoded_columns"] = cat_cols
        else:  # label encoding (default)
            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.log["encoded_columns"].append(col)

        self.df = df
        return self

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------
    def scale_features(self, method="minmax", columns=None):
        df = self.df
        numeric_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return self

        scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.log["scaled_columns"] = numeric_cols

        self.df = df
        return self

    # ------------------------------------------------------------------
    def run_full_pipeline(self, encode=True, scale=True, encoding_method="label",
                           scaling_method="minmax", remove_outlier_rows=True):
        """Run the complete preprocessing pipeline in the documented order."""
        self.handle_missing_values()
        self.remove_duplicates()
        if remove_outlier_rows:
            self.remove_outliers()
        self.clean_data()
        if encode:
            self.encode_categorical(method=encoding_method)
        if scale:
            self.scale_features(method=scaling_method)
        return self.df

    def get_log(self):
        return self.log
