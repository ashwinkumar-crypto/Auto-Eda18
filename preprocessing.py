"""
preprocessing.py
----------------
Automated data preprocessing engine for AutoEDA Pro.

Handles:
  - Missing value imputation (median for numeric, mode for categorical
    by default; see missing_values.py for advanced techniques)
  - Duplicate row removal
  - Outlier detection & removal using the IQR method by default (see
    outliers.py for Z-Score / Isolation Forest / LOF / DBSCAN)
  - Categorical encoding (Label Encoding / optional One-Hot Encoding;
    see encoding.py for Ordinal / Frequency / Target encoding)
  - Feature scaling (Min-Max Normalization / Standardization; see
    scaling.py for Robust / MaxAbs / Normalizer)
  - General cleaning (constant columns, dtype fixes)

`run_full_pipeline()` preserves the original, simple preprocessing
behavior. `run_advanced_pipeline()` is an opt-in superset that lets
callers choose any supported technique per stage and records a full
preprocessing summary (see get_preprocessing_summary()).
"""

import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from utils import is_numeric_column, is_categorical_column, is_datetime_like
from missing_values import MissingValueHandler
from outliers import OutlierDetector
from encoding import CategoricalEncoder
from scaling import FeatureScaler

logger = logging.getLogger("autoeda.preprocessing")


class DataPreprocessor:
    def __init__(self, df):
        self.original_df = df.copy()
        self.df = df.copy()
        self.log = {
            "missing_before": {},
            "missing_filled": {},
            "missing_technique": None,
            "duplicates_removed": 0,
            "outliers_removed": 0,
            "outlier_method": None,
            "constant_columns_dropped": [],
            "encoded_columns": [],
            "encoding_method": None,
            "scaled_columns": [],
            "scaling_method": None,
            "steps_applied": [],
        }

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------
    def handle_missing_values(self):
        """Fill missing values: median for numeric columns, mode for
        categorical columns. See missing_values.py for advanced techniques
        (KNN, MICE, forward/backward fill, etc.) exposed via
        run_advanced_pipeline()."""
        df = self.df
        self.log["missing_before"] = df.isnull().sum().to_dict()

        for col in df.columns:
            try:
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
            except Exception:
                logger.exception("Failed to impute missing values for column '%s'", col)

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
        cat_cols = columns or [c for c in df.columns if is_categorical_column(df[c])
                                and not is_datetime_like(df[c], name_hint=c)]

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
        """Run the complete preprocessing pipeline in the documented order.
        Preserved for backward compatibility with existing callers."""
        try:
            self.handle_missing_values()
            self.remove_duplicates()
            if remove_outlier_rows:
                self.remove_outliers()
            self.clean_data()
            if encode:
                self.encode_categorical(method=encoding_method)
            if scale:
                self.scale_features(method=scaling_method)
        except Exception:
            logger.exception("run_full_pipeline failed")
            raise
        return self.df

    def get_log(self):
        return self.log

    # ------------------------------------------------------------------
    # Advanced pipeline: any missing-value / outlier / encoding / scaling
    # technique from missing_values.py, outliers.py, encoding.py and
    # scaling.py, plus a full step-by-step summary for reporting.
    # ------------------------------------------------------------------
    def run_advanced_pipeline(self, target_column=None,
                               missing_technique="auto", knn_neighbors=5,
                               remove_outlier_rows=True, outlier_method="auto",
                               encode=True, encoding_method="auto",
                               scale=True, scaling_method="auto",
                               drop_duplicates=True):
        """
        Run preprocessing with explicit (or auto-recommended) techniques
        for each stage, recording a full summary for the dashboard/report.

        Any of missing_technique / outlier_method / encoding_method /
        scaling_method may be "auto" to use the corresponding module's
        recommendation heuristic.
        """
        df = self.df
        summary_rows = []

        # -- Missing values --------------------------------------------------
        mv_handler = MissingValueHandler(df)
        missing_before = int(df.isnull().sum().sum())
        self.log["missing_before"] = df.isnull().sum().to_dict()
        chosen_missing = mv_handler.recommend_technique() if missing_technique == "auto" else missing_technique
        if chosen_missing != "none":
            try:
                df = mv_handler.apply(chosen_missing, knn_neighbors=knn_neighbors)
            except Exception:
                logger.exception("Missing-value technique '%s' failed; falling back to median/mode.", chosen_missing)
                df = mv_handler.apply("median")
                chosen_missing = "median (fallback)"
        missing_after = int(df.isnull().sum().sum())
        self.log["missing_technique"] = chosen_missing
        self.log["missing_filled"] = {"count": missing_before - missing_after}
        summary_rows.append(("Missing Values", chosen_missing,
                              f"{missing_before - missing_after} values filled/removed"))
        self.log["steps_applied"].append("missing_values")

        # -- Duplicates --------------------------------------------------
        if drop_duplicates:
            before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            removed = before - len(df)
            self.log["duplicates_removed"] = removed
            summary_rows.append(("Duplicates", "drop_duplicates", f"{removed} rows removed"))
            self.log["steps_applied"].append("duplicates")

        # -- Constant column cleanup --------------------------------------
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if constant_cols:
            df = df.drop(columns=constant_cols)
            self.log["constant_columns_dropped"] = constant_cols
            summary_rows.append(("Constant Columns", "drop", ", ".join(constant_cols)))
        self.log["steps_applied"].append("clean_constant_columns")

        # -- Outliers --------------------------------------------------
        if remove_outlier_rows:
            detector = OutlierDetector(df)
            chosen_outlier = "iqr" if outlier_method == "auto" else outlier_method
            try:
                result = detector.detect(chosen_outlier)
                before = len(df)
                df = df[result["mask"]].reset_index(drop=True)
                self.log["outliers_removed"] = before - len(df)
                self.log["outlier_method"] = chosen_outlier
                summary_rows.append(("Outliers", chosen_outlier, f"{before - len(df)} rows removed ({result['percent']}%)"))
            except Exception:
                logger.exception("Outlier method '%s' failed; skipping outlier removal.", chosen_outlier)
            self.log["steps_applied"].append("outliers")

        self.original_df = df.copy()  # snapshot pre-encoding/scaling, for readable stats/exports

        # -- Encoding --------------------------------------------------
        if encode:
            encoder = CategoricalEncoder(df)
            chosen_encoding = encoder.recommend_method(target_column=target_column) if encoding_method == "auto" else encoding_method
            if chosen_encoding != "none":
                try:
                    df = encoder.encode(method=chosen_encoding, target_column=target_column)
                    self.log["encoded_columns"] = encoder.log.get("columns_encoded", [])
                    self.log["encoding_method"] = chosen_encoding
                    summary_rows.append(("Encoding", chosen_encoding, ", ".join(self.log["encoded_columns"]) or "-"))
                except Exception:
                    logger.exception("Encoding method '%s' failed; falling back to label encoding.", chosen_encoding)
                    encoder = CategoricalEncoder(df)
                    df = encoder.encode(method="label")
                    self.log["encoding_method"] = "label (fallback)"
            self.log["steps_applied"].append("encoding")

        # -- Scaling --------------------------------------------------
        if scale:
            scaler = FeatureScaler(df)
            chosen_scaling = scaler.recommend_method() if scaling_method == "auto" else scaling_method
            if chosen_scaling != "none":
                df = scaler.scale(method=chosen_scaling)
                self.log["scaled_columns"] = scaler.log.get("columns_scaled", [])
                self.log["scaling_method"] = chosen_scaling
                summary_rows.append(("Scaling", chosen_scaling, ", ".join(self.log["scaled_columns"]) or "-"))
            self.log["steps_applied"].append("scaling")

        self.df = df
        self._advanced_summary_rows = summary_rows
        return self.df

    def get_preprocessing_summary(self):
        """Return a DataFrame summarizing each step run by run_advanced_pipeline()."""
        rows = getattr(self, "_advanced_summary_rows", None)
        if not rows:
            return pd.DataFrame(columns=["Step", "Technique", "Detail"])
        return pd.DataFrame(rows, columns=["Step", "Technique", "Detail"])
