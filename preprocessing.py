"""
preprocessing.py
----------------
Automated data preprocessing engine for AutoEDA Pro.

Handles:
  - Missing value imputation: median/mode (default), mean/mode, KNN
    imputation, or row/column dropping — selectable per run
  - Duplicate row removal
  - Outlier detection & removal using the IQR method
  - Categorical encoding (Label Encoding / optional One-Hot Encoding)
  - Feature scaling (Min-Max Normalization / Standardization)
  - General cleaning (constant columns, dtype fixes)
  - Defensive error handling around every step, logged to self.log["errors"]
    instead of silently crashing the pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer

from utils import is_numeric_column, is_categorical_column


class DataPreprocessor:
    def __init__(self, df):
        self.original_df = df.copy()
        self.df = df.copy()
        self.log = {
            "missing_before": {},
            "missing_filled": {},
            "missing_method": None,
            "rows_dropped_missing": 0,
            "columns_dropped_missing": [],
            "duplicates_removed": 0,
            "outliers_removed": 0,
            "constant_columns_dropped": [],
            "encoded_columns": [],
            "scaled_columns": [],
            "errors": [],
        }

    def _log_error(self, step, exc):
        """Record a step failure without crashing the whole pipeline."""
        message = f"{step}: {type(exc).__name__} - {exc}"
        self.log["errors"].append(message)

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------
    def handle_missing_values(self, method="median_mode", knn_neighbors=5,
                               drop_column_threshold=0.5):
        """
        method:
          - "median_mode" (default): median for numeric, mode for categorical
          - "mean_mode": mean for numeric, mode for categorical
          - "knn": KNN imputation for numeric columns (mode for categorical)
          - "drop_rows": drop any row containing a missing value
          - "drop_columns": drop columns whose missing fraction exceeds
            drop_column_threshold, then median/mode-impute what remains
        """
        df = self.df
        self.log["missing_method"] = method
        try:
            self.log["missing_before"] = df.isnull().sum().to_dict()
        except Exception as e:
            self._log_error("handle_missing_values.count_missing", e)
            self.log["missing_before"] = {}

        try:
            if method == "drop_rows":
                before = len(df)
                df = df.dropna().reset_index(drop=True)
                self.log["rows_dropped_missing"] = before - len(df)

            elif method == "drop_columns":
                frac_missing = df.isnull().mean()
                cols_to_drop = frac_missing[frac_missing > drop_column_threshold].index.tolist()
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                    self.log["columns_dropped_missing"] = cols_to_drop
                df = self._impute_median_mode(df)

            elif method == "mean_mode":
                df = self._impute_mean_mode(df)

            elif method == "knn":
                df = self._impute_knn(df, n_neighbors=knn_neighbors)

            else:  # "median_mode" (default / fallback for unknown methods)
                df = self._impute_median_mode(df)

        except Exception as e:
            self._log_error(f"handle_missing_values.{method}", e)
            # Fall back to the safest option so the pipeline can continue.
            try:
                df = self._impute_median_mode(df)
            except Exception as e2:
                self._log_error("handle_missing_values.fallback", e2)

        self.df = df
        return self

    def _impute_median_mode(self, df):
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            try:
                if is_numeric_column(df[col]):
                    fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)
                    self.log["missing_filled"][col] = f"median ({fill_value:.4f})"
                else:
                    mode_series = df[col].mode(dropna=True)
                    fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                    df[col] = df[col].fillna(fill_value)
                    self.log["missing_filled"][col] = f"mode ({fill_value})"
            except Exception as e:
                self._log_error(f"impute_median_mode.{col}", e)
        return df

    def _impute_mean_mode(self, df):
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            try:
                if is_numeric_column(df[col]):
                    fill_value = df[col].mean()
                    df[col] = df[col].fillna(fill_value)
                    self.log["missing_filled"][col] = f"mean ({fill_value:.4f})"
                else:
                    mode_series = df[col].mode(dropna=True)
                    fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                    df[col] = df[col].fillna(fill_value)
                    self.log["missing_filled"][col] = f"mode ({fill_value})"
            except Exception as e:
                self._log_error(f"impute_mean_mode.{col}", e)
        return df

    def _impute_knn(self, df, n_neighbors=5):
        numeric_cols = [c for c in df.columns if is_numeric_column(df[c]) and df[c].isnull().any()]
        cat_cols = [c for c in df.columns if not is_numeric_column(df[c]) and df[c].isnull().any()]

        if numeric_cols:
            try:
                imputer = KNNImputer(n_neighbors=n_neighbors)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                for col in numeric_cols:
                    self.log["missing_filled"][col] = f"KNN (k={n_neighbors})"
            except Exception as e:
                self._log_error("impute_knn.numeric", e)
                # Fall back to median for numeric columns if KNN fails
                for col in numeric_cols:
                    fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)
                    self.log["missing_filled"][col] = f"median fallback ({fill_value:.4f})"

        for col in cat_cols:
            try:
                mode_series = df[col].mode(dropna=True)
                fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                df[col] = df[col].fillna(fill_value)
                self.log["missing_filled"][col] = f"mode ({fill_value})"
            except Exception as e:
                self._log_error(f"impute_knn.categorical.{col}", e)

        return df

    # ------------------------------------------------------------------
    # Duplicates
    # ------------------------------------------------------------------
    def remove_duplicates(self):
        try:
            before = len(self.df)
            self.df = self.df.drop_duplicates().reset_index(drop=True)
            self.log["duplicates_removed"] = before - len(self.df)
        except Exception as e:
            self._log_error("remove_duplicates", e)
        return self

    # ------------------------------------------------------------------
    # Outliers (IQR method)
    # ------------------------------------------------------------------
    def remove_outliers(self, columns=None, factor=1.5):
        try:
            df = self.df
            numeric_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

            before = len(df)
            mask = pd.Series(True, index=df.index)

            for col in numeric_cols:
                try:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - factor * iqr
                    upper = q3 + factor * iqr
                    mask &= df[col].between(lower, upper)
                except Exception as e:
                    self._log_error(f"remove_outliers.{col}", e)

            self.df = df[mask].reset_index(drop=True)
            self.log["outliers_removed"] = before - len(self.df)
        except Exception as e:
            self._log_error("remove_outliers", e)
        return self

    # ------------------------------------------------------------------
    # Cleaning: constant columns, dtype fixes
    # ------------------------------------------------------------------
    def clean_data(self):
        try:
            df = self.df
            constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
            if constant_cols:
                df = df.drop(columns=constant_cols)
                self.log["constant_columns_dropped"] = constant_cols
            self.df = df
        except Exception as e:
            self._log_error("clean_data", e)
        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode_categorical(self, method="label", columns=None):
        try:
            df = self.df
            cat_cols = columns or [c for c in df.columns if is_categorical_column(df[c])]

            if method == "onehot":
                if cat_cols:
                    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
                    self.log["encoded_columns"] = cat_cols
            else:  # label encoding (default)
                for col in cat_cols:
                    try:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                        self.log["encoded_columns"].append(col)
                    except Exception as e:
                        self._log_error(f"encode_categorical.{col}", e)

            self.df = df
        except Exception as e:
            self._log_error("encode_categorical", e)
        return self

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------
    def scale_features(self, method="minmax", columns=None):
        try:
            df = self.df
            numeric_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_cols:
                return self

            scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.log["scaled_columns"] = numeric_cols

            self.df = df
        except Exception as e:
            self._log_error("scale_features", e)
        return self

    # ------------------------------------------------------------------
    def run_full_pipeline(self, encode=True, scale=True, encoding_method="label",
                           scaling_method="minmax", remove_outlier_rows=True,
                           missing_value_method="median_mode", knn_neighbors=5,
                           drop_column_threshold=0.5):
        """Run the complete preprocessing pipeline in the documented order.

        missing_value_method: "median_mode" | "mean_mode" | "knn" |
                               "drop_rows" | "drop_columns"
        """
        self.handle_missing_values(
            method=missing_value_method,
            knn_neighbors=knn_neighbors,
            drop_column_threshold=drop_column_threshold,
        )
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

    def has_errors(self):
        return len(self.log["errors"]) > 0
