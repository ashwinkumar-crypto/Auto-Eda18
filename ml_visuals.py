"""
ml_visuals.py
--------------
Visualization helpers for the ML module: confusion matrix heatmap, ROC
curve (binary classification), residual plot (regression), a feature
importance bar chart (top-N), and an optional SHAP summary plot (used
only if the `shap` package is installed; skipped gracefully otherwise).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger("autoeda.ml_visuals")

try:
    import shap  # noqa: F401
    _HAS_SHAP = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_SHAP = False


def _save(fig, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
def plot_confusion_matrix(cm, labels, output_dir="charts", filename="confusion_matrix.png") -> str:
    """Save a confusion matrix heatmap. `cm` may be a list of lists or ndarray."""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.9), max(4, len(labels) * 0.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return _save(fig, output_dir, filename)


# ----------------------------------------------------------------------
def plot_roc_curve(y_test, y_proba, output_dir="charts", filename="roc_curve.png") -> Optional[str]:
    """
    Save an ROC curve. Only meaningful for binary classification with
    predicted probabilities; for multiclass, plots one-vs-rest curves
    for up to 5 classes. Returns None if probabilities are unavailable.
    """
    if y_proba is None:
        return None

    y_test = np.asarray(y_test)
    classes = np.unique(y_test)
    fig, ax = plt.subplots(figsize=(6, 6))

    try:
        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color="#2e86de")
        else:
            for i, cls in enumerate(classes[:5]):
                fpr, tpr, _ = roc_curve((y_test == cls).astype(int), y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"Class {cls} (AUC = {roc_auc:.3f})")
    except Exception as exc:
        logger.warning("Could not compute ROC curve: %s", exc)
        plt.close(fig)
        return None

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return _save(fig, output_dir, filename)


# ----------------------------------------------------------------------
def plot_residuals(y_test, y_pred, output_dir="charts", filename="residual_plot.png") -> str:
    """Save a residual plot (predicted vs. residuals) for regression diagnostics."""
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(y_pred, residuals, alpha=0.6, color="#4C72B0", edgecolor="none")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs. Predicted")

    sns.histplot(residuals, kde=True, ax=axes[1], color="#DD8452")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")

    fig.tight_layout()
    return _save(fig, output_dir, filename)


# ----------------------------------------------------------------------
def plot_feature_importance(importance_df: pd.DataFrame, output_dir="charts",
                             filename="feature_importance_chart.png", top_n: int = 10,
                             score_column: str = "composite_score",
                             name_column: str = "feature") -> Optional[str]:
    """Save a horizontal bar chart of the top-N most important features."""
    if importance_df is None or importance_df.empty:
        return None
    top = importance_df.sort_values(score_column, ascending=False).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7, max(4, len(top) * 0.5)))
    ax.barh(top[name_column].astype(str), top[score_column], color="#2e86de")
    ax.set_xlabel(score_column.replace("_", " ").title())
    ax.set_title(f"Top {len(top)} Most Important Features")
    fig.tight_layout()
    return _save(fig, output_dir, filename)


# ----------------------------------------------------------------------
def plot_shap_summary(model, X_sample: pd.DataFrame, output_dir="charts",
                       filename="shap_summary.png", max_samples: int = 200) -> Optional[str]:
    """
    Save a SHAP summary plot, if the optional `shap` package is
    installed and the model is SHAP-compatible. Returns None otherwise
    (this is a soft, best-effort feature, not a hard requirement).
    """
    if not _HAS_SHAP:
        logger.info("shap is not installed; skipping SHAP summary plot.")
        return None

    try:
        sample = X_sample.sample(min(max_samples, len(X_sample)), random_state=42)
        explainer = shap.Explainer(model, sample)
        try:
            shap_values = explainer(sample)
        except Exception as inner_exc:
            # Tree explainers can fail their internal additivity check on some
            # sklearn tree models/versions; retry with the check disabled.
            logger.info("Retrying SHAP explanation with check_additivity=False: %s", inner_exc)
            shap_values = explainer(sample, check_additivity=False)

        fig = plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, sample, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        return _save(fig, output_dir, filename)
    except Exception as exc:
        logger.warning("SHAP summary plot could not be generated: %s", exc)
        plt.close("all")
        return None
