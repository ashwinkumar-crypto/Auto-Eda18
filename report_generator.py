"""
report_generator.py
--------------------
Generates:
  - Automatic Insights (observations about the dataset)
  - Recommendations (guidance on ML-readiness / next steps)
  - insights.txt output file
  - profiling_report.html using fg-data-profiling (falls back to
    ydata-profiling / a lightweight custom profile if unavailable)
"""

import os
import numpy as np


def generate_insights(original_df, cleaned_df, eda_overview, correlation_results, preprocess_log):
    """Build a list of automatic, human-readable insight strings."""
    insights = []

    rows, cols = cleaned_df.shape
    insights.append(f"Dataset contains {rows:,} rows and {cols} columns after cleaning.")

    missing_total = eda_overview["missing_values_total"]
    if missing_total == 0:
        insights.append("No missing values detected in the cleaned dataset.")
    else:
        insights.append(
            f"{missing_total} missing values were detected and imputed "
            f"(median for numeric columns, mode for categorical columns)."
        )

    dup_removed = preprocess_log.get("duplicates_removed", 0)
    if dup_removed > 0:
        insights.append(f"{dup_removed} duplicate records were found and removed successfully.")
    else:
        insights.append("No duplicate records were found in the dataset.")

    outliers_removed = preprocess_log.get("outliers_removed", 0)
    if outliers_removed > 0:
        insights.append(
            f"{outliers_removed} outlier rows were detected via the IQR method and removed."
        )
    else:
        insights.append("No significant outliers were detected using the IQR method.")

    strongest_pos = correlation_results.get("strongest_positive")
    strongest_neg = correlation_results.get("strongest_negative")
    if strongest_pos:
        (a, b), val = strongest_pos
        insights.append(f"Strong positive correlation found between {a} and {b} (r = {val:.2f}).")
    if strongest_neg:
        (a, b), val = strongest_neg
        if val < -0.3:
            insights.append(f"Notable negative correlation found between {a} and {b} (r = {val:.2f}).")

    if preprocess_log.get("scaled_columns"):
        insights.append("Numerical features have been normalized/standardized for ML readiness.")

    if preprocess_log.get("encoded_columns"):
        insights.append(
            f"Categorical columns ({', '.join(preprocess_log['encoded_columns'])}) were encoded numerically."
        )

    if preprocess_log.get("constant_columns_dropped"):
        insights.append(
            f"Constant columns removed as they carry no information: "
            f"{', '.join(preprocess_log['constant_columns_dropped'])}."
        )

    insights.append("Dataset is clean and suitable for machine learning.")
    return insights


def generate_recommendations(cleaned_df, correlation_results, eda_overview):
    """Build a list of recommendation strings based on dataset characteristics."""
    recs = []

    if eda_overview["missing_values_total"] == 0:
        recs.append("Dataset quality is excellent. No further preprocessing required.")
    else:
        recs.append("Consider reviewing imputed values for columns with high missingness.")

    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        recs.append("Suitable for Regression models given multiple numeric features.")

    cat_cols = cleaned_df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) >= 1 or len(numeric_cols) >= 1:
        recs.append("Suitable for Classification models if a target label is defined.")

    if cleaned_df.shape[1] > 15:
        recs.append("Additional feature selection may improve model performance given the high dimensionality.")
    else:
        recs.append("Additional feature engineering may improve performance.")

    if cleaned_df.shape[0] < 500:
        recs.append("Dataset is relatively small; consider cross-validation to ensure robust model evaluation.")

    return recs


def save_insights_file(insights, recommendations, path="output/insights.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ["AUTOEDA PRO - AUTOMATIC INSIGHTS", "=" * 40, ""]
    lines.append("INSIGHTS:")
    for i, item in enumerate(insights, 1):
        lines.append(f"{i}. {item}")
    lines.append("")
    lines.append("RECOMMENDATIONS:")
    for i, item in enumerate(recommendations, 1):
        lines.append(f"{i}. {item}")

    content = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def generate_profiling_report(df, output_path="output/profiling_report.html", title="AutoEDA Pro - Profiling Report"):
    """
    Generate a full profiling report using fg-data-profiling
    (falls back to ydata-profiling, then a minimal built-in profile
    if neither package is available).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        from data_profiling import ProfileReport  # fg-data-profiling
        profile = ProfileReport(df, title=title, explorative=True)
        profile.to_file(output_path)
        return output_path
    except Exception:
        pass

    try:
        from ydata_profiling import ProfileReport  # legacy fallback
        profile = ProfileReport(df, title=title, explorative=True)
        profile.to_file(output_path)
        return output_path
    except Exception:
        pass

    # Minimal built-in fallback profile (no external profiling package available)
    html = _minimal_profile_html(df, title)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def _minimal_profile_html(df, title):
    """A lightweight profiling HTML fallback, used only if no profiling
    library could be loaded."""
    desc = df.describe(include="all").fillna("").to_html(classes="table")
    dtypes = df.dtypes.astype(str).to_frame("dtype").to_html(classes="table")
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; background:#f7f8fa; }}
h1 {{ color:#2c3e50; }}
.table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; background:#fff; }}
.table th, .table td {{ border: 1px solid #ddd; padding: 6px 10px; font-size: 13px; }}
.table th {{ background:#2c3e50; color:#fff; }}
</style></head>
<body>
<h1>{title}</h1>
<p>Rows: {df.shape[0]} &nbsp;|&nbsp; Columns: {df.shape[1]}</p>
<h2>Column Types</h2>
{dtypes}
<h2>Descriptive Statistics</h2>
{desc}
</body></html>"""
