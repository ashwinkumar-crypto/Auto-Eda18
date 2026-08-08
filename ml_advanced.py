"""
ml_advanced.py
---------------
Extended machine-learning module for AutoEDA Pro, layered on top of the
existing model_training.py ("ML Lab"). Adds:

  - Automatic task detection: classification / regression / clustering
  - A larger model zoo: Logistic Regression, Decision Tree, Random
    Forest, XGBoost, LightGBM, KNN, Naive Bayes, SVM (classification);
    Linear Regression, Decision Tree, Random Forest, XGBoost
    (regression); K-Means, DBSCAN, Agglomerative Clustering (clustering)
  - Full evaluation metrics per task type
  - GridSearchCV / RandomizedSearchCV hyperparameter tuning with
    cross-validation

model_training.py's train_and_evaluate() (single model, used by the
Streamlit "ML Lab" tab) is untouched; this module is additive and is
used for multi-model comparison (model_metrics.csv) and tuning.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)

logger = logging.getLogger("autoeda.ml_advanced")

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGBOOST = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LIGHTGBM = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CATBOOST = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_CATBOOST = False


# ----------------------------------------------------------------------
# Model registries
# ----------------------------------------------------------------------
def _classification_registry() -> dict:
    registry = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
        "DecisionTree": lambda: DecisionTreeClassifier(random_state=42),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": lambda: KNeighborsClassifier(),
        "NaiveBayes": lambda: GaussianNB(),
        "SVM": lambda: SVC(probability=True, random_state=42),
    }
    if _HAS_XGBOOST:
        registry["XGBoost"] = lambda: XGBClassifier(
            n_estimators=200, eval_metric="logloss", random_state=42, verbosity=0)
    if _HAS_LIGHTGBM:
        registry["LightGBM"] = lambda: LGBMClassifier(n_estimators=200, random_state=42, verbosity=-1)
    if _HAS_CATBOOST:
        registry["CatBoost"] = lambda: CatBoostClassifier(iterations=200, random_state=42, verbose=False)
    return registry


def _regression_registry() -> dict:
    registry = {
        "LinearRegression": lambda: LinearRegression(),
        "DecisionTree": lambda: DecisionTreeRegressor(random_state=42),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=200, random_state=42),
    }
    if _HAS_XGBOOST:
        registry["XGBoost"] = lambda: XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    if _HAS_LIGHTGBM:
        registry["LightGBM"] = lambda: LGBMRegressor(n_estimators=200, random_state=42, verbosity=-1)
    if _HAS_CATBOOST:
        registry["CatBoost"] = lambda: CatBoostRegressor(iterations=200, random_state=42, verbose=False)
    return registry


CLUSTERING_ALGORITHMS = ["kmeans", "dbscan", "agglomerative"]

PARAM_GRIDS = {
    "RandomForest": {"n_estimators": [100, 200, 400], "max_depth": [None, 5, 10, 20]},
    "DecisionTree": {"max_depth": [None, 3, 5, 10], "min_samples_split": [2, 5, 10]},
    "LogisticRegression": {"C": [0.01, 0.1, 1, 10]},
    "KNN": {"n_neighbors": [3, 5, 7, 11]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "LinearRegression": {},
    "XGBoost": {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]},
}


# ----------------------------------------------------------------------
def detect_task_type(df: pd.DataFrame, target_column: Optional[str] = None) -> str:
    """Return 'classification', 'regression', or 'clustering' (no target)."""
    if not target_column or target_column not in df.columns:
        return "clustering"
    series = df[target_column]
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique(dropna=True)
        if n_unique > 20 and n_unique / max(len(series), 1) > 0.05:
            return "regression"
    return "classification"


def _prepare_features(df: pd.DataFrame, target_column: Optional[str] = None):
    feature_df = df.drop(columns=[target_column]) if target_column else df.copy()
    feature_df = feature_df.select_dtypes(include=[np.number])
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
    return feature_df


def specificity_score(y_true, y_pred) -> float:
    """
    Specificity (true negative rate). For binary problems this is the
    standard TN / (TN + FP). For multiclass problems, returns the
    macro-average of per-class specificity (each class treated as
    positive vs. all others in turn).
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    specificities = []
    total = cm.sum()
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specificities))


# ----------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------
def compare_classification_models(df: pd.DataFrame, target_column: str,
                                   algorithms: Optional[list] = None,
                                   test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    """Train & evaluate several classifiers, returning a ranked comparison table."""
    registry = _classification_registry()
    algorithms = algorithms or list(registry.keys())

    work_df = df.dropna(subset=[target_column])
    X = _prepare_features(work_df, target_column)
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns available for classification.")

    y_raw = work_df[target_column]
    if not pd.api.types.is_numeric_dtype(y_raw):
        y = LabelEncoder().fit_transform(y_raw.astype(str))
    else:
        y = y_raw.values
    n_classes = len(set(y))
    average = "binary" if n_classes == 2 else "weighted"

    stratify = y if n_classes > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    rows = []
    for name in algorithms:
        if name not in registry:
            continue
        try:
            model = registry[name]()
            train_start = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - train_start

            predict_start = time.perf_counter()
            y_pred = model.predict(X_test)
            predict_time = time.perf_counter() - predict_start

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
                "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
                "f1": f1_score(y_test, y_pred, average=average, zero_division=0),
                "specificity": specificity_score(y_test, y_pred),
                "mcc": matthews_corrcoef(y_test, y_pred),
                "cohen_kappa": cohen_kappa_score(y_test, y_pred),
                "train_time_sec": train_time,
                "predict_time_sec": predict_time,
            }
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_test)
                    if n_classes == 2:
                        metrics["roc_auc"] = roc_auc_score(y_test, proba[:, 1])
                        metrics["pr_auc"] = average_precision_score(y_test, proba[:, 1])
                        metrics["log_loss"] = log_loss(y_test, proba)
                    else:
                        metrics["roc_auc"] = roc_auc_score(y_test, proba, multi_class="ovr")
                        metrics["log_loss"] = log_loss(y_test, proba)
                except Exception:
                    pass

            rows.append({"model": name, **{k: round(float(v), 4) for k, v in metrics.items()}})
        except Exception as exc:
            logger.warning("Classifier '%s' failed: %s", name, exc)

    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        comparison = comparison.sort_values("f1", ascending=False).reset_index(drop=True)
    return comparison


def detailed_classification_result(df: pd.DataFrame, target_column: str, algorithm: str,
                                    test_size: float = 0.2, random_state: int = 42) -> dict:
    """Full detail for one classifier: confusion matrix + text classification report."""
    registry = _classification_registry()
    if algorithm not in registry:
        raise ValueError(f"Unknown classification algorithm '{algorithm}'.")

    work_df = df.dropna(subset=[target_column])
    X = _prepare_features(work_df, target_column)
    y_raw = work_df[target_column]
    label_encoder = None
    if not pd.api.types.is_numeric_dtype(y_raw):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        labels = list(label_encoder.classes_)
    else:
        y = y_raw.values
        labels = sorted(pd.unique(y_raw))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if len(set(y)) > 1 else None)

    model = registry[algorithm]()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "labels": [str(l) for l in labels],
        "classification_report": classification_report(
            y_test, y_pred, target_names=[str(l) for l in labels], zero_division=0),
        "specificity": specificity_score(y_test, y_pred),
    }


def get_classification_diagnostics(df: pd.DataFrame, target_column: str, algorithm: str,
                                    test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Fit one classifier and return everything needed to plot a confusion
    matrix and ROC curve, and to compute per-class evaluation metrics:
    y_test, y_pred, predicted probabilities (if available), and labels.
    """
    registry = _classification_registry()
    if algorithm not in registry:
        raise ValueError(f"Unknown classification algorithm '{algorithm}'.")

    work_df = df.dropna(subset=[target_column])
    X = _prepare_features(work_df, target_column)
    y_raw = work_df[target_column]
    if not pd.api.types.is_numeric_dtype(y_raw):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        labels = list(label_encoder.classes_)
    else:
        y = y_raw.values
        labels = sorted(pd.unique(y_raw))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if len(set(y)) > 1 else None)

    model = registry[algorithm]()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

    return {
        "model": model,
        "algorithm": algorithm,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "labels": [str(l) for l in labels],
        "n_classes": len(set(y)),
        "evaluation_metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, average="binary" if len(set(y)) == 2 else "weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, average="binary" if len(set(y)) == 2 else "weighted", zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, average="binary" if len(set(y)) == 2 else "weighted", zero_division=0)), 4),
            "specificity": round(specificity_score(y_test, y_pred), 4),
            "mcc": round(float(matthews_corrcoef(y_test, y_pred)), 4),
            "cohen_kappa": round(float(cohen_kappa_score(y_test, y_pred)), 4),
        },
    }


def get_regression_diagnostics(df: pd.DataFrame, target_column: str, algorithm: str,
                                test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Fit one regressor and return everything needed to plot residuals and
    compute detailed evaluation metrics: y_test, y_pred, and metrics.
    """
    registry = _regression_registry()
    if algorithm not in registry:
        raise ValueError(f"Unknown regression algorithm '{algorithm}'.")

    work_df = df.dropna(subset=[target_column])
    X = _prepare_features(work_df, target_column)
    y = work_df[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    model = registry[algorithm]()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    n, p = X_test.shape[0], X_test.shape[1]
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)
    mape = mean_absolute_percentage_error(y_test, y_pred) if np.all(y_test != 0) else float("nan")

    return {
        "model": model,
        "algorithm": algorithm,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "evaluation_metrics": {
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "mse": round(float(mean_squared_error(y_test, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            "r2": round(float(r2), 4),
            "adjusted_r2": round(float(adj_r2), 4),
            "mape": round(float(mape), 4) if not np.isnan(mape) else None,
            "explained_variance": round(float(explained_variance_score(y_test, y_pred)), 4),
        },
    }


# ----------------------------------------------------------------------
# Regression
# ----------------------------------------------------------------------
def compare_regression_models(df: pd.DataFrame, target_column: str,
                               algorithms: Optional[list] = None,
                               test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    """Train & evaluate several regressors, returning a ranked comparison table."""
    registry = _regression_registry()
    algorithms = algorithms or list(registry.keys())

    work_df = df.dropna(subset=[target_column])
    X = _prepare_features(work_df, target_column)
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns available for regression.")
    y = work_df[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    rows = []
    n, p = X_test.shape[0], X_test.shape[1]
    for name in algorithms:
        if name not in registry:
            continue
        try:
            model = registry[name]()
            train_start = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - train_start

            predict_start = time.perf_counter()
            y_pred = model.predict(X_test)
            predict_time = time.perf_counter() - predict_start

            r2 = r2_score(y_test, y_pred)
            adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)
            mape = mean_absolute_percentage_error(y_test, y_pred) if np.all(y_test != 0) else np.nan

            metrics = {
                "mae": mean_absolute_error(y_test, y_pred),
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2": r2,
                "adjusted_r2": adj_r2,
                "mape": mape,
                "explained_variance": explained_variance_score(y_test, y_pred),
                "train_time_sec": train_time,
                "predict_time_sec": predict_time,
            }
            rows.append({"model": name, **{k: round(float(v), 4) if pd.notna(v) else None for k, v in metrics.items()}})
        except Exception as exc:
            logger.warning("Regressor '%s' failed: %s", name, exc)

    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        comparison = comparison.sort_values("r2", ascending=False).reset_index(drop=True)
    return comparison


# ----------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------
def run_clustering(df: pd.DataFrame, algorithm: str = "kmeans", n_clusters: int = 3,
                    columns: Optional[list] = None, **kwargs) -> dict:
    """Fit a clustering algorithm and return labels + internal validation metrics."""
    from sklearn.preprocessing import StandardScaler

    numeric_df = df[columns] if columns else df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.fillna(numeric_df.median())
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric columns available for clustering.")

    X = StandardScaler().fit_transform(numeric_df)

    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)
    elif algorithm == "dbscan":
        model = DBSCAN(eps=kwargs.get("eps", 1.5), min_samples=kwargs.get("min_samples", 5))
        labels = model.fit_predict(X)
    elif algorithm == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Unknown clustering algorithm '{algorithm}'. Choose from {CLUSTERING_ALGORITHMS}.")

    n_found = len(set(labels)) - (1 if -1 in labels else 0)
    metrics = {}
    if n_found >= 2:
        try:
            metrics["silhouette_score"] = float(silhouette_score(X, labels))
            metrics["davies_bouldin_index"] = float(davies_bouldin_score(X, labels))
            metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X, labels))
        except Exception as exc:
            logger.warning("Could not compute clustering metrics: %s", exc)

    return {
        "algorithm": algorithm,
        "n_clusters_found": n_found,
        "labels": labels.tolist(),
        "metrics": metrics,
        "columns_used": numeric_df.columns.tolist(),
    }


# ----------------------------------------------------------------------
# Hyperparameter tuning
# ----------------------------------------------------------------------
def tune_hyperparameters(df: pd.DataFrame, target_column: str, algorithm: str,
                          task_type: str = "classification", search: str = "grid",
                          cv: int = 3, n_iter: int = 10, random_state: int = 42) -> dict:
    """
    Run GridSearchCV or RandomizedSearchCV with cross-validation for a
    given model and return the best parameters, best CV score, and the
    fitted estimator's held-out test score.
    """
    registry = _classification_registry() if task_type == "classification" else _regression_registry()
    if algorithm not in registry:
        raise ValueError(f"Unknown algorithm '{algorithm}' for task type '{task_type}'.")
    param_grid = PARAM_GRIDS.get(algorithm, {})
    if not param_grid:
        raise ValueError(f"No hyperparameter grid defined for '{algorithm}'.")

    work_df = df.dropna(subset=[target_column])
    X = _prepare_features(work_df, target_column)
    y_raw = work_df[target_column]
    if task_type == "classification" and not pd.api.types.is_numeric_dtype(y_raw):
        y = LabelEncoder().fit_transform(y_raw.astype(str))
    else:
        y = y_raw.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state,
        stratify=y if task_type == "classification" and len(set(y)) > 1 else None)

    base_model = registry[algorithm]()
    scoring = "f1_weighted" if task_type == "classification" else "r2"

    if search == "random":
        searcher = RandomizedSearchCV(base_model, param_grid, n_iter=min(n_iter, _grid_size(param_grid)),
                                       cv=cv, scoring=scoring, random_state=random_state, n_jobs=-1)
    else:
        searcher = GridSearchCV(base_model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)

    searcher.fit(X_train, y_train)
    test_score = searcher.score(X_test, y_test)

    return {
        "algorithm": algorithm,
        "search_type": search,
        "best_params": searcher.best_params_,
        "best_cv_score": round(float(searcher.best_score_), 4),
        "test_score": round(float(test_score), 4),
        "scoring": scoring,
    }


def _grid_size(param_grid: dict) -> int:
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return max(size, 1)
