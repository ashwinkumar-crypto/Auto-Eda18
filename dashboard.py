"""
dashboard.py
------------
Builds the interactive HTML dashboard (output/dashboard.html) combining
dataset overview, data quality score, preprocessing summary, statistics,
missing values, duplicate/outlier analysis, chart images, feature
importance, model comparison, automatic insights, and recommendations
into one page using Jinja2.
"""

import os
import base64
import pandas as pd
from jinja2 import Template

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{{ title }}</title>
<style>
  :root {
    --navy: #1f2d3d;
    --accent: #2e86de;
    --bg: #f4f6f9;
    --card: #ffffff;
    --border: #e3e7ee;
  }
  * { box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: var(--bg);
    margin: 0;
    color: #2c3e50;
  }
  header {
    background: linear-gradient(135deg, var(--navy), #34495e);
    color: #fff;
    padding: 32px 40px;
  }
  header h1 { margin: 0; font-size: 28px; }
  header p { margin: 6px 0 0; opacity: 0.85; }
  .container { max-width: 1200px; margin: 0 auto; padding: 30px 40px 60px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 30px; }
  .stat-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 18px 20px; text-align: center;
  }
  .stat-card .value { font-size: 26px; font-weight: 700; color: var(--accent); }
  .stat-card .label { font-size: 13px; color: #7f8c9a; margin-top: 4px; }
  .stat-card .value.excellent, .stat-card .value.good { color: #27ae60; }
  .stat-card .value.fair { color: #e67e22; }
  .stat-card .value.poor, .stat-card .value.critical { color: #e74c3c; }
  .progress-track { width: 100%; height: 14px; background: #eef1f6; border-radius: 7px; overflow: hidden; }
  .progress-fill { height: 100%; border-radius: 7px; transition: width 0.3s ease; }
  .progress-fill.excellent, .progress-fill.good { background: #27ae60; }
  .progress-fill.fair { background: #e67e22; }
  .progress-fill.poor, .progress-fill.critical { background: #e74c3c; }
  section { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 24px 28px; margin-bottom: 24px; }
  section h2 { margin-top: 0; font-size: 19px; color: var(--navy); border-bottom: 2px solid var(--border); padding-bottom: 10px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { border: 1px solid var(--border); padding: 8px 10px; text-align: left; }
  th { background: #eef1f6; }
  .chart-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
  .chart-card img { width: 100%; border-radius: 6px; border: 1px solid var(--border); }
  .chart-card figcaption { text-align: center; font-size: 13px; margin-top: 6px; color: #7f8c9a; }
  ul.insight-list li { margin-bottom: 8px; line-height: 1.5; }
  .badge { display:inline-block; background:#e8f4fd; color:var(--accent); border-radius:5px;
    padding:2px 8px; font-size:12px; margin-left:8px; }
  footer { text-align:center; color:#95a5a6; font-size:12px; padding:20px; }
</style>
</head>
<body>
<header>
  <h1>AutoEDA Pro Dashboard</h1>
  <p>{{ dataset_name }} &middot; Generated automatically</p>
</header>
<div class="container">

  <div class="grid">
    <div class="stat-card"><div class="value">{{ overview.rows }}</div><div class="label">Rows</div></div>
    <div class="stat-card"><div class="value">{{ overview.columns }}</div><div class="label">Columns</div></div>
    <div class="stat-card"><div class="value">{{ overview.memory_usage }}</div><div class="label">Memory Usage</div></div>
    {% if quality_report %}
    <div class="stat-card">
      <div class="value {{ quality_report.quality_grade | lower }}">{{ quality_report.quality_score }}/100</div>
      <div class="label">Data Quality &middot; {{ quality_report.quality_grade }}</div>
    </div>
    {% else %}
    <div class="stat-card"><div class="value">{{ overview.missing_values_total }}</div><div class="label">Missing Values</div></div>
    {% endif %}
  </div>

  <section>
    <h2>Dataset Overview</h2>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Rows</td><td>{{ overview.rows }}</td></tr>
      <tr><td>Columns</td><td>{{ overview.columns }}</td></tr>
      <tr><td>Memory Usage</td><td>{{ overview.memory_usage }}</td></tr>
      <tr><td>Missing Values (total)</td><td>{{ overview.missing_values_total }}</td></tr>
      <tr><td>Duplicate Rows (original)</td><td>{{ overview.duplicate_rows }}</td></tr>
    </table>
  </section>

  {% if quality_report %}
  <section>
    <h2>Data Quality Report <span class="badge">{{ quality_report.quality_grade }}</span></h2>
    <div style="display:flex; align-items:center; gap:28px; flex-wrap:wrap; margin-bottom:18px;">
      <div style="flex-shrink:0;">{{ quality_gauge_svg | safe }}</div>
      <div style="flex:1; min-width:260px;">
        <div class="progress-track">
          <div class="progress-fill {{ quality_report.quality_grade | lower }}" style="width: {{ quality_report.quality_score }}%;"></div>
        </div>
        <p style="margin:8px 0 0; font-size:13px; color:#7f8c9a;">
          Overall quality score of <strong>{{ quality_report.quality_score }}/100</strong>, based on missingness,
          duplication, constant/zero-variance columns, high-cardinality columns, and correlated feature pairs.
        </p>
      </div>
    </div>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Missing Values</td><td>{{ quality_report.missing.total_missing }} ({{ quality_report.missing.missing_percent }}%)</td></tr>
      <tr><td>Duplicate Rows</td><td>{{ quality_report.duplicate.duplicate_count }} ({{ quality_report.duplicate.duplicate_percent }}%)</td></tr>
      <tr><td>Constant Columns</td><td>{{ quality_report.constant_columns | join(', ') if quality_report.constant_columns else 'None' }}</td></tr>
      <tr><td>Zero-Variance Columns</td><td>{{ quality_report.zero_variance_columns | join(', ') if quality_report.zero_variance_columns else 'None' }}</td></tr>
      <tr><td>High-Cardinality Columns</td><td>{{ quality_report.high_cardinality_columns | join(', ') if quality_report.high_cardinality_columns else 'None' }}</td></tr>
      <tr><td>Highly Correlated Pairs</td>
        <td>{% if quality_report.highly_correlated_pairs %}{% for a, b, v in quality_report.highly_correlated_pairs %}{{ a }} &harr; {{ b }} (r={{ v }}){% if not loop.last %}, {% endif %}{% endfor %}{% else %}None{% endif %}</td></tr>
      {% if quality_report.class_distribution %}
      <tr><td>Target Class Balance</td><td>Imbalance ratio &asymp; {{ quality_report.class_distribution.imbalance_ratio }}:1
        {% if quality_report.class_distribution.is_imbalanced %} <span class="badge">Imbalanced</span>{% endif %}</td></tr>
      {% endif %}
    </table>
  </section>

  {% endif %}

  {% if preprocessing_table %}
  <section>
    <h2>Preprocessing Summary</h2>
    {{ preprocessing_table | safe }}
  </section>
  {% endif %}

  <section>
    <h2>Summary Statistics</h2>
    {{ stats_table | safe }}
  </section>

  <section>
    <h2>Missing Value Analysis</h2>
    {% if missing_table %}
      {{ missing_table | safe }}
    {% else %}
      <p>No missing values remain in the cleaned dataset.</p>
    {% endif %}
  </section>

  <section>
    <h2>Duplicate Analysis</h2>
    <p>Duplicate rows found in original data: <strong>{{ duplicate_info.duplicate_count }}</strong>
       ({{ duplicate_info.duplicate_percent }}%)</p>
  </section>

  {% if outlier_table %}
  <section>
    <h2>Outlier Analysis</h2>
    {{ outlier_table | safe }}
  </section>
  {% endif %}

  <section>
    <h2>Visualizations</h2>
    <div class="chart-grid">
      {% for chart in charts %}
      <figure class="chart-card">
        <img src="data:image/png;base64,{{ chart.data }}" alt="{{ chart.name }}">
        <figcaption>{{ chart.name }}</figcaption>
      </figure>
      {% endfor %}
    </div>
  </section>

  {% if feature_importance_table %}
  <section>
    <h2>Feature Importance</h2>
    {% if feature_importance_chart %}
    <img src="data:image/png;base64,{{ feature_importance_chart }}" alt="Feature Importance" style="max-width:100%; border-radius:6px; margin-bottom:16px;">
    {% endif %}
    {{ feature_importance_table | safe }}
  </section>
  {% endif %}

  {% if shap_summary_chart %}
  <section>
    <h2>SHAP Value Plot</h2>
    <img src="data:image/png;base64,{{ shap_summary_chart }}" alt="SHAP Summary" style="max-width:100%; border-radius:6px;">
  </section>
  {% endif %}

  {% if model_comparison_table %}
  <section>
    <h2>Model Comparison{% if model_task_type %} <span class="badge">{{ model_task_type }}</span>{% endif %}</h2>
    {{ model_comparison_table | safe }}
  </section>
  {% endif %}

  {% if confusion_matrix_chart or roc_curve_chart %}
  <section>
    <h2>Classification Diagnostics{% if best_model_name %} <span class="badge">Best model: {{ best_model_name }}</span>{% endif %}</h2>
    <div class="chart-grid">
      {% if confusion_matrix_chart %}
      <figure class="chart-card">
        <img src="data:image/png;base64,{{ confusion_matrix_chart }}" alt="Confusion Matrix">
        <figcaption>Confusion Matrix</figcaption>
      </figure>
      {% endif %}
      {% if roc_curve_chart %}
      <figure class="chart-card">
        <img src="data:image/png;base64,{{ roc_curve_chart }}" alt="ROC Curve">
        <figcaption>ROC Curve</figcaption>
      </figure>
      {% endif %}
    </div>
    {% if classification_report_text %}
    <pre style="background:#f8f9fb; padding:14px; border-radius:6px; overflow-x:auto; font-size:12px;">{{ classification_report_text }}</pre>
    {% endif %}
  </section>
  {% endif %}

  {% if residual_plot_chart %}
  <section>
    <h2>Residual Analysis{% if best_model_name %} <span class="badge">Best model: {{ best_model_name }}</span>{% endif %}</h2>
    <img src="data:image/png;base64,{{ residual_plot_chart }}" alt="Residual Plot" style="max-width:100%; border-radius:6px;">
  </section>
  {% endif %}

  {% if clustering_info %}
  <section>
    <h2>Clustering Results</h2>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Algorithm</td><td>{{ clustering_info.algorithm }}</td></tr>
      <tr><td>Clusters found</td><td>{{ clustering_info.n_clusters_found }}</td></tr>
      {% for k, v in clustering_info.metrics.items() %}
      <tr><td>{{ k }}</td><td>{{ "%.4f"|format(v) }}</td></tr>
      {% endfor %}
    </table>
  </section>
  {% endif %}

  {% if tuning_info %}
  <section>
    <h2>Hyperparameter Tuning</h2>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Algorithm</td><td>{{ tuning_info.algorithm }}</td></tr>
      <tr><td>Search type</td><td>{{ tuning_info.search_type }}</td></tr>
      <tr><td>Best params</td><td>{{ tuning_info.best_params }}</td></tr>
      <tr><td>Best CV score ({{ tuning_info.scoring }})</td><td>{{ tuning_info.best_cv_score }}</td></tr>
      <tr><td>Held-out test score</td><td>{{ tuning_info.test_score }}</td></tr>
    </table>
  </section>
  {% endif %}

  <section>
    <h2>Automatic Insights</h2>
    <ul class="insight-list">
      {% for insight in insights %}<li>{{ insight }}</li>{% endfor %}
    </ul>
  </section>

  <section>
    <h2>Recommendations</h2>
    <ul class="insight-list">
      {% for rec in recommendations %}<li>{{ rec }}</li>{% endfor %}
    </ul>
  </section>

  <section>
    <h2>Final Conclusion</h2>
    <p>{{ conclusion }}</p>
  </section>

</div>
<footer>Generated by AutoEDA Pro</footer>
</body>
</html>
"""


def _encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _quality_gauge_svg(score, grade, size=110):
    """Build a small inline SVG donut gauge for the data quality score."""
    color_map = {
        "excellent": "#27ae60", "good": "#27ae60",
        "fair": "#e67e22", "poor": "#e74c3c", "critical": "#e74c3c",
    }
    color = color_map.get(str(grade).lower(), "#2e86de")
    radius = 45
    circumference = 2 * 3.14159265 * radius
    offset = circumference * (1 - max(0, min(100, score)) / 100)
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 120 120">
  <circle cx="60" cy="60" r="{radius}" fill="none" stroke="#eef1f6" stroke-width="12"/>
  <circle cx="60" cy="60" r="{radius}" fill="none" stroke="{color}" stroke-width="12"
    stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}"
    stroke-linecap="round" transform="rotate(-90 60 60)"/>
  <text x="60" y="56" text-anchor="middle" font-size="24" font-weight="700" fill="{color}" font-family="Segoe UI, Arial, sans-serif">{score}</text>
  <text x="60" y="74" text-anchor="middle" font-size="11" fill="#7f8c9a" font-family="Segoe UI, Arial, sans-serif">/ 100</text>
</svg>
""".strip()


def build_dashboard(
    dataset_name,
    overview,
    stats_df,
    missing_df,
    duplicate_info,
    chart_paths,
    insights,
    recommendations,
    conclusion,
    output_path="output/dashboard.html",
    quality_report=None,
    preprocessing_summary_df=None,
    outlier_summary_df=None,
    feature_importance_df=None,
    model_comparison_df=None,
    model_task_type=None,
    clustering_result=None,
    tuning_result=None,
    confusion_matrix_path=None,
    roc_curve_path=None,
    residual_plot_path=None,
    feature_importance_chart_path=None,
    shap_summary_path=None,
    classification_report_text=None,
    best_model_name=None,
):
    """
    Build and save the full HTML dashboard. All arguments after
    `output_path` are optional so this remains backward-compatible with
    the original (pre-upgrade) call signature.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    stats_table = (
        stats_df.to_html(classes="table", border=0) if stats_df is not None and not stats_df.empty
        else "<p>No numeric columns available.</p>"
    )
    missing_table = (
        missing_df.to_html(classes="table", border=0) if missing_df is not None and not missing_df.empty else None
    )
    preprocessing_table = (
        preprocessing_summary_df.to_html(classes="table", border=0, index=False)
        if preprocessing_summary_df is not None and not preprocessing_summary_df.empty else None
    )
    outlier_table = (
        outlier_summary_df.to_html(classes="table", border=0, index=False)
        if outlier_summary_df is not None and not outlier_summary_df.empty else None
    )
    feature_importance_table = (
        feature_importance_df.to_html(classes="table", border=0, index=False)
        if feature_importance_df is not None and not feature_importance_df.empty else None
    )
    model_comparison_table = (
        model_comparison_df.to_html(classes="table", border=0, index=False)
        if model_comparison_df is not None and not model_comparison_df.empty else None
    )

    quality_gauge_svg = (
        _quality_gauge_svg(quality_report["quality_score"], quality_report["quality_grade"])
        if quality_report else ""
    )

    charts = []
    for path in chart_paths:
        if path and os.path.exists(path):
            charts.append({"name": os.path.splitext(os.path.basename(path))[0].title(), "data": _encode_image(path)})

    confusion_matrix_chart = _encode_image(confusion_matrix_path) if confusion_matrix_path and os.path.exists(confusion_matrix_path) else None
    roc_curve_chart = _encode_image(roc_curve_path) if roc_curve_path and os.path.exists(roc_curve_path) else None
    residual_plot_chart = _encode_image(residual_plot_path) if residual_plot_path and os.path.exists(residual_plot_path) else None
    feature_importance_chart = _encode_image(feature_importance_chart_path) if feature_importance_chart_path and os.path.exists(feature_importance_chart_path) else None
    shap_summary_chart = _encode_image(shap_summary_path) if shap_summary_path and os.path.exists(shap_summary_path) else None

    template = Template(DASHBOARD_TEMPLATE)
    html = template.render(
        title=f"AutoEDA Pro - {dataset_name}",
        dataset_name=dataset_name,
        overview=overview,
        stats_table=stats_table,
        missing_table=missing_table,
        duplicate_info=duplicate_info,
        charts=charts,
        insights=insights,
        recommendations=recommendations,
        conclusion=conclusion,
        quality_report=quality_report,
        quality_gauge_svg=quality_gauge_svg,
        preprocessing_table=preprocessing_table,
        outlier_table=outlier_table,
        feature_importance_table=feature_importance_table,
        model_comparison_table=model_comparison_table,
        model_task_type=model_task_type,
        clustering_info=clustering_result,
        tuning_info=tuning_result,
        confusion_matrix_chart=confusion_matrix_chart,
        roc_curve_chart=roc_curve_chart,
        residual_plot_chart=residual_plot_chart,
        feature_importance_chart=feature_importance_chart,
        shap_summary_chart=shap_summary_chart,
        classification_report_text=classification_report_text,
        best_model_name=best_model_name,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
