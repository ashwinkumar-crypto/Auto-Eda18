"""
model_training.py
------------------
A lightweight "ML Lab" for AutoEDA Pro — inspired by the ML Lab panel in
the AI Workbench UI mockup (dataset preview/auto-EDA, algorithm picker,
train/test split, metrics, confusion matrix, feature importance, and a
"code that ran" panel). Unlike the mockup, this trains a REAL
scikit-learn model on the user's actual (cleaned) dataset.

Supports:
  - Automatic task-type detection (classification vs regression) based
    on the target column's dtype and cardinality, with manual override.
  - Classification: RandomForest, LogisticRegression, KNN, SVC
  - Regression: RandomForest, LinearRegression, KNN, SVR
  - Confusion matrix (classification) / actual-vs-predicted (regression)
  - Feature importances (tree-based models) or coefficients (linear models)
  - A dynamically generated code snippet mirroring exactly what ran
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, r2_score, mean_absolute_error, mean_squared_error,
)

CLASSIFICATION_MODELS = {
    "RandomForestClassifier": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "KNeighborsClassifier": KNeighborsClassifier,
    "SVC": SVC,
}

REGRESSION_MODELS = {
    "RandomForestRegressor": RandomForestRegressor,
    "LinearRegression": LinearRegression,
    "KNeighborsRegressor": KNeighborsRegressor,
    "SVR": SVR,
}


def suggest_task_type(series):
    """Heuristic: numeric with many unique values -> regression,
    otherwise classification."""
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique(dropna=True)
        if n_unique > 20 and n_unique / max(len(series), 1) > 0.05:
            return "regression"
    return "classification"


def _build_model(task_type, algorithm, **kwargs):
    registry = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
    model_cls = registry[algorithm]

    if algorithm.startswith("RandomForest"):
        return model_cls(n_estimators=200, random_state=42)
    if algorithm == "LogisticRegression":
        return model_cls(max_iter=1000)
    return model_cls()


def train_and_evaluate(df, target_column, task_type="auto", algorithm=None, test_size=0.2, random_state=42):
    """
    Train a scikit-learn model on `df` predicting `target_column`.

    Returns a dict with: task_type, algorithm, metrics, confusion_matrix
    (classification only, with labels), predictions_vs_actual (regression
    only), feature_importance (dict or None), and generated_code (str).
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    work_df = df.dropna(subset=[target_column]).copy()
    y_raw = work_df[target_column]

    if task_type == "auto":
        task_type = suggest_task_type(y_raw)

    if algorithm is None:
        algorithm = "RandomForestClassifier" if task_type == "classification" else "RandomForestRegressor"

    # Build feature matrix: numeric columns only, excluding the target.
    feature_df = work_df.drop(columns=[target_column])
    feature_df = feature_df.select_dtypes(include=[np.number])
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    if feature_df.shape[1] == 0:
        raise ValueError(
            "No numeric feature columns are available to train on. "
            "Try encoding categorical columns first (see preprocessing options)."
        )

    label_encoder = None
    class_labels = None
    if task_type == "classification":
        if not pd.api.types.is_numeric_dtype(y_raw):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_raw.astype(str))
            class_labels = list(label_encoder.classes_)
        else:
            y = y_raw.values
            class_labels = sorted(pd.unique(y_raw))
    else:
        y = y_raw.values

    X = feature_df.values
    feature_names = feature_df.columns.tolist()

    stratify = y if (task_type == "classification" and len(set(y)) > 1) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    model = _build_model(task_type, algorithm)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result = {
        "task_type": task_type,
        "algorithm": algorithm,
        "target_column": target_column,
        "feature_names": feature_names,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    if task_type == "classification":
        average = "binary" if len(set(y)) == 2 else "weighted"
        result["metrics"] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average=average, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average=average, zero_division=0)),
        }
        cm = confusion_matrix(y_test, y_pred)
        result["confusion_matrix"] = cm.tolist()
        result["class_labels"] = [str(c) for c in class_labels] if class_labels is not None else None
    else:
        result["metrics"] = {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        }
        result["predictions_vs_actual"] = {
            "actual": [float(v) for v in y_test[:200]],
            "predicted": [float(v) for v in y_pred[:200]],
        }

    # Feature importance / coefficients, if the model exposes them.
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = dict(zip(feature_names, [float(v) for v in model.feature_importances_]))
    elif hasattr(model, "coef_"):
        coefs = model.coef_
        coefs = coefs[0] if getattr(coefs, "ndim", 1) > 1 else coefs
        importance = dict(zip(feature_names, [float(abs(v)) for v in coefs]))

    if importance:
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    result["feature_importance"] = importance

    result["generated_code"] = _generate_code(task_type, algorithm, target_column, feature_names, test_size)

    return result


def _generate_code(task_type, algorithm, target_column, feature_names, test_size):
    import_line = {
        "RandomForestClassifier": "from sklearn.ensemble import RandomForestClassifier",
        "LogisticRegression": "from sklearn.linear_model import LogisticRegression",
        "KNeighborsClassifier": "from sklearn.neighbors import KNeighborsClassifier",
        "SVC": "from sklearn.svm import SVC",
        "RandomForestRegressor": "from sklearn.ensemble import RandomForestRegressor",
        "LinearRegression": "from sklearn.linear_model import LinearRegression",
        "KNeighborsRegressor": "from sklearn.neighbors import KNeighborsRegressor",
        "SVR": "from sklearn.svm import SVR",
    }[algorithm]

    init_args = "n_estimators=200, random_state=42" if algorithm.startswith("RandomForest") else (
        "max_iter=1000" if algorithm == "LogisticRegression" else ""
    )

    metric_lines = (
        "acc = accuracy_score(y_test, y_pred)\n"
        "prec = precision_score(y_test, y_pred, average='weighted')\n"
        "rec = recall_score(y_test, y_pred, average='weighted')"
        if task_type == "classification" else
        "r2 = r2_score(y_test, y_pred)\n"
        "mae = mean_absolute_error(y_test, y_pred)"
    )

    metrics_import = (
        "from sklearn.metrics import accuracy_score, precision_score, recall_score"
        if task_type == "classification" else
        "from sklearn.metrics import r2_score, mean_absolute_error"
    )

    features_repr = ", ".join(f'"{f}"' for f in feature_names[:6])
    if len(feature_names) > 6:
        features_repr += ", ..."

    return f"""from sklearn.model_selection import train_test_split
{import_line}
{metrics_import}

feature_cols = [{features_repr}]
X = df[feature_cols]
y = df["{target_column}"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state=42
)

model = {algorithm}({init_args})
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

{metric_lines}
"""
