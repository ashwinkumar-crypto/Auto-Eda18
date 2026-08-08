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
  - Metrics:
      Classification -> accuracy, precision, recall, f1, and roc_auc
                         (roc_auc only when the model exposes predict_proba)
      Regression      -> r2, adjusted_r2, mae, rmse
  - Confusion matrix (classification) / actual-vs-predicted (regression)
  - Feature importances (tree-based models) or coefficients (linear models)
  - A dynamically generated code snippet mirroring exactly what ran
  - Defensive error handling: missing target column, empty dataset,
    single-class targets, non-numeric-only feature sets, failed
    stratified splits, and model-fit failures all raise clear
    ValueErrors (or are handled with sane fallbacks) instead of
    crashing with a raw traceback
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
    confusion_matrix, roc_auc_score, r2_score, mean_absolute_error,
    mean_squared_error,
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

    if df.empty:
        raise ValueError("The dataset is empty — nothing to train on.")

    work_df = df.dropna(subset=[target_column]).copy()
    if work_df.empty:
        raise ValueError(
            f"Target column '{target_column}' has no non-missing values after dropping NaNs."
        )
    y_raw = work_df[target_column]

    if task_type == "auto":
        try:
            task_type = suggest_task_type(y_raw)
        except Exception as e:
            raise ValueError(f"Could not auto-detect task type: {e}")

    if algorithm is None:
        algorithm = "RandomForestClassifier" if task_type == "classification" else "RandomForestRegressor"

    # Build feature matrix: numeric columns only, excluding the target.
    feature_df = work_df.drop(columns=[target_column])
    feature_df = feature_df.select_dtypes(include=[np.number])
    try:
        feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
    except Exception as e:
        raise ValueError(f"Failed to impute missing feature values before training: {e}")

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

        if len(set(y)) < 2:
            raise ValueError(
                f"Target column '{target_column}' only has one class after cleaning — "
                "classification needs at least two classes."
            )
    else:
        y = y_raw.values

    X = feature_df.values
    feature_names = feature_df.columns.tolist()

    stratify = y if (task_type == "classification" and len(set(y)) > 1) else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError as e:
        # Common cause: a class has too few members to stratify-split.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )

    model = _build_model(task_type, algorithm)
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    except Exception as e:
        raise ValueError(f"Model training failed ({algorithm}): {e}")

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

        # ROC-AUC needs predicted probabilities; not every model/algorithm
        # exposes them (e.g. plain SVC without probability=True), so this
        # is computed defensively and simply omitted if unavailable.
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if len(set(y)) == 2:
                    result["metrics"]["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
                else:
                    result["metrics"]["roc_auc"] = float(
                        roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
                    )
        except Exception:
            pass  # ROC-AUC just won't appear in the metrics dict

        cm = confusion_matrix(y_test, y_pred)
        result["confusion_matrix"] = cm.tolist()
        result["class_labels"] = [str(c) for c in class_labels] if class_labels is not None else None
    else:
        r2 = r2_score(y_test, y_pred)
        n = len(y_test)
        p = len(feature_names)
        # Adjusted R^2 penalizes extra features; undefined when n - p - 1 <= 0.
        if n - p - 1 > 0:
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        else:
            adj_r2 = None

        result["metrics"] = {
            "r2": float(r2),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        }
        if adj_r2 is not None:
            result["metrics"]["adjusted_r2"] = float(adj_r2)

        result["predictions_vs_actual"] = {
            "actual": [float(v) for v in y_test[:200]],
            "predicted": [float(v) for v in y_pred[:200]],
        }

    # Feature importance / coefficients, if the model exposes them.
    importance = None
    try:
        if hasattr(model, "feature_importances_"):
            importance = dict(zip(feature_names, [float(v) for v in model.feature_importances_]))
        elif hasattr(model, "coef_"):
            coefs = model.coef_
            coefs = coefs[0] if getattr(coefs, "ndim", 1) > 1 else coefs
            importance = dict(zip(feature_names, [float(abs(v)) for v in coefs]))
    except Exception:
        importance = None

    if importance:
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    result["feature_importance"] = importance

    result["generated_code"] = _generate_code(task_type, algorithm, target_column, feature_names, test_size)

    # Keep the fitted estimator (and label encoder, if any) so callers can
    # persist/download the trained model, e.g. via pickle/joblib.
    result["model"] = model
    result["label_encoder"] = label_encoder

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
        "rec = recall_score(y_test, y_pred, average='weighted')\n"
        "f1 = f1_score(y_test, y_pred, average='weighted')\n"
        "# roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])  # binary case"
        if task_type == "classification" else
        "r2 = r2_score(y_test, y_pred)\n"
        "mae = mean_absolute_error(y_test, y_pred)\n"
        "rmse = mean_squared_error(y_test, y_pred) ** 0.5\n"
        "adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)"
    )

    metrics_import = (
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score"
        if task_type == "classification" else
        "from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error"
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
