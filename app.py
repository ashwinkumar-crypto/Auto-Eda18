"""
app.py
------
Streamlit web interface for AutoEDA Pro.

A dark, card-based "control room" dashboard: top navbar with logo/branding
and a one-click summary ZIP, a sectioned pipeline-options sidebar, and a
tabbed workspace (Charts, Insights & Recommendations, Cleaned Data,
Feature Lab, Full Dashboard, ML Lab, and — when a target column is set —
Model Comparison).

Run with:
    streamlit run app.py
"""

import html
import io
import os
import pickle
import tempfile
import zipfile

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from main import run_pipeline
from utils import SUPPORTED_EXTENSIONS, load_dataset
from model_training import train_and_evaluate, CLASSIFICATION_MODELS, REGRESSION_MODELS, suggest_task_type

st.set_page_config(page_title="AutoEDA Pro", page_icon="📊", layout="wide")

# ----------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

PALETTES = {
    "dark": dict(
        bg="#0a0e1a", panel="#121729", panel_alt="#161c33", sidebar="#0d1220",
        border="rgba(255,255,255,0.09)", text="#e7e9f3", muted="#8b93ab",
        accent="#7c6cf3", accent_soft="rgba(124,108,243,0.16)", accent_text="#ffffff",
        success="#35d68a", success_soft="rgba(53,214,138,0.15)",
        danger="#ff5c7a", danger_soft="rgba(255,92,122,0.15)",
        input_bg="#1a2036", chip_bg="#1a2036", shadow="0 8px 24px rgba(0,0,0,0.35)",
    ),
    "light": dict(
        bg="#f4f5fb", panel="#ffffff", panel_alt="#f7f8fd", sidebar="#ffffff",
        border="#e6e8f4", text="#1c2033", muted="#6b7086",
        accent="#6c5ce7", accent_soft="rgba(108,92,231,0.10)", accent_text="#ffffff",
        success="#16a34a", success_soft="rgba(22,163,74,0.12)",
        danger="#e0345c", danger_soft="rgba(224,52,92,0.10)",
        input_bg="#f2f3fa", chip_bg="#eef0fa", shadow="0 8px 24px rgba(30,34,70,0.08)",
    ),
}


def inject_css(p):
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', sans-serif; }}
        .stApp {{ background: {p['bg']}; color: {p['text']}; }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: {p['sidebar']}; border-right: 1px solid {p['border']};
        }}
        section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
        .sb-label {{
            text-transform: uppercase; letter-spacing: .07em; font-size: 11px;
            font-weight: 700; color: {p['muted']}; margin: 18px 0 6px 0;
        }}
        .sb-label:first-of-type {{ margin-top: 4px; }}

        /* Cards */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: {p['panel']}; border: 1px solid {p['border']} !important;
            border-radius: 14px !important; box-shadow: {p['shadow']};
        }}

        /* Buttons */
        .stButton > button, .stDownloadButton > button {{
            border-radius: 10px; font-weight: 600; border: 1px solid {p['border']};
            background: {p['panel_alt']}; color: {p['text']}; transition: all .15s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: {p['accent']}; color: {p['accent']};
        }}
        .stButton > button[kind="primary"] {{
            background: {p['accent']}; color: {p['accent_text']}; border: none;
        }}
        .stButton > button[kind="primary"]:hover {{ opacity: .9; color: {p['accent_text']}; }}
        .stButton > button:disabled {{ opacity: .45; }}

        /* Inputs */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div,
        .stNumberInput input {{
            background: {p['input_bg']} !important; color: {p['text']} !important;
            border-radius: 8px !important; border: 1px solid {p['border']} !important;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {p['border']}; }}
        .stTabs [data-baseweb="tab"] {{
            color: {p['muted']}; font-weight: 600; padding: 10px 14px;
        }}
        .stTabs [aria-selected="true"] {{ color: {p['accent']} !important; }}
        .stTabs [data-baseweb="tab-highlight"] {{ background-color: {p['accent']} !important; }}

        /* Alerts */
        div[data-testid="stAlert"] {{ border-radius: 10px; }}

        /* File uploader */
        section[data-testid="stFileUploaderDropzone"] {{
            background: {p['panel_alt']}; border: 1.5px dashed {p['border']}; border-radius: 12px;
        }}

        /* Navbar */
        .aeda-navbar-wrap div[data-testid="stVerticalBlockBorderWrapper"] {{
            padding: 2px 6px;
        }}
        .aeda-logo-row {{ display: flex; align-items: center; gap: 10px; height: 100%; }}
        .aeda-logo-icon {{
            width: 34px; height: 34px; border-radius: 9px; flex-shrink: 0;
            background: linear-gradient(135deg, #4f8cff, {p['accent']});
            display: flex; align-items: center; justify-content: center; font-size: 17px;
        }}
        .aeda-logo-title {{ font-weight: 800; font-size: 17px; letter-spacing: -.01em; color: {p['text']}; }}

        /* Stat / metric cards */
        .aeda-stats {{ display: flex; flex-direction: column; gap: 12px; }}
        .aeda-stats.horizontal {{ flex-direction: row; }}
        .aeda-stat-card {{
            flex: 1; background: {p['panel_alt']}; border: 1px solid {p['border']};
            border-radius: 12px; padding: 14px 16px;
        }}
        .aeda-stat-value {{ font-size: 22px; font-weight: 800; color: {p['text']}; line-height: 1.2; }}
        .aeda-stat-label {{ font-size: 12px; color: {p['muted']}; margin-top: 2px; }}
        .aeda-badge {{
            display: inline-block; margin-top: 8px; padding: 2px 10px; border-radius: 999px;
            font-size: 11px; font-weight: 700; background: {p['success_soft']}; color: {p['success']};
        }}

        /* Insight / recommendation lists */
        .aeda-list-title {{
            font-size: 12px; font-weight: 800; letter-spacing: .06em; color: {p['muted']};
            margin-bottom: 10px; text-transform: uppercase;
        }}
        .aeda-list ol {{ margin: 0; padding-left: 20px; color: {p['text']}; }}
        .aeda-list li {{ margin-bottom: 8px; line-height: 1.5; font-size: 14.5px; }}
        .aeda-list li::marker {{ color: {p['accent']}; font-weight: 700; }}

        div[data-testid="stImage"] img {{ border-radius: 10px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


palette = PALETTES[st.session_state.theme]
inject_css(palette)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
PROFILING_ROW_LIMIT = 50_000
DOWNLOAD_SPEC_KEYS = [
    ("cleaned_csv", "cleaned_data.csv"),
    ("dashboard", "dashboard.html"),
    ("profiling_report", "profiling_report.html"),
    ("insights", "insights.txt"),
    ("feature_engineered_csv", "feature_engineered.csv"),
    ("model_comparison_csv", "model_comparison.csv"),
    ("model_metrics_csv", "model_metrics.csv"),
    ("evaluation_metrics_csv", "evaluation_metrics.csv"),
    ("preprocessing_summary_csv", "preprocessing_summary.csv"),
]


def build_summary_zip(results):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key, default_name in DOWNLOAD_SPEC_KEYS:
            path = results.get(key)
            if path and os.path.exists(path):
                zf.write(path, arcname=os.path.basename(path) or default_name)
    buf.seek(0)
    return buf


def render_stat_cards(items, horizontal=False):
    cls = "aeda-stats horizontal" if horizontal else "aeda-stats"
    parts = [f"<div class='{cls}'>"]
    for item in items:
        badge = f"<div class='aeda-badge'>{html.escape(str(item['badge']))}</div>" if item.get("badge") else ""
        parts.append(
            "<div class='aeda-stat-card'>"
            f"<div class='aeda-stat-value'>{html.escape(str(item['value']))}</div>"
            f"<div class='aeda-stat-label'>{html.escape(str(item['label']))}</div>"
            f"{badge}</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_numbered_list(title, items):
    parts = [f"<div class='aeda-list'><div class='aeda-list-title'>{html.escape(title)}</div><ol>"]
    for item in items:
        parts.append(f"<li>{html.escape(item)}</li>")
    parts.append("</ol></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def parse_insights_file(text):
    """Split the generated insights.txt back into (insights, recommendations) lists."""
    insights, recs = [], []
    bucket = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("INSIGHTS:"):
            bucket = insights
            continue
        if stripped.upper().startswith("RECOMMENDATIONS:"):
            bucket = recs
            continue
        if not stripped or stripped.startswith("=") or bucket is None:
            continue
        # Strip a leading "N. " numbering if present.
        parts = stripped.split(". ", 1)
        bucket.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else stripped)
    return insights, recs


def sb_label(text):
    st.sidebar.markdown(f"<div class='sb-label'>{html.escape(text)}</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Top navbar
# ----------------------------------------------------------------------
st.markdown("<div class='aeda-navbar-wrap'>", unsafe_allow_html=True)
with st.container(border=True):
    nav_logo, nav_spacer, nav_zip, nav_deploy, nav_theme = st.columns([4, 3, 2, 1.4, 0.7], vertical_alignment="center")
    with nav_logo:
        st.markdown(
            "<div class='aeda-logo-row'>"
            "<div class='aeda-logo-icon'>📊</div>"
            "<div class='aeda-logo-title'>AutoEDA Pro</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with nav_zip:
        _results_for_nav = st.session_state.get("results")
        if _results_for_nav:
            st.download_button(
                "⬇ Download summary ZIP",
                build_summary_zip(_results_for_nav),
                file_name="autoeda_pro_summary.zip",
                mime="application/zip",
                key="dl_summary_zip",
            )
        else:
            st.button("⬇ Download summary ZIP", disabled=True, key="dl_summary_zip_disabled",
                       help="Run the pipeline on a dataset first.")
    with nav_deploy:
        st.button("🚀 Deploy", type="primary", disabled=True, key="deploy_btn",
                   help="Deployment isn't available in this environment.")
    with nav_theme:
        icon = "🌙" if st.session_state.theme == "light" else "☀️"
        if st.button(icon, key="theme_toggle", help="Toggle light / dark mode"):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Sidebar - options
# ----------------------------------------------------------------------
st.sidebar.markdown("<div class='sb-label' style='margin-top:0;'>Pipeline Options</div>", unsafe_allow_html=True)

advanced_mode = st.sidebar.toggle(
    "Use advanced pipeline", value=True,
    help="Adds a Data Quality Score, auto-recommended techniques per stage, "
         "extended charts, feature importance, and multi-model comparison. "
         "Turn off to reproduce the original simple pipeline exactly.",
)

target_column_input = st.sidebar.text_input(
    "Target column (optional)", placeholder="e.g. price, target, class",
    help="Enables the Data Quality class-imbalance check, Feature Importance, "
         "and Model Comparison tabs. Leave blank to skip these.",
)

sb_label("Missing values")
missing_technique = st.sidebar.selectbox(
    "Technique", ["auto", "median", "mean", "mode", "constant", "ffill", "bfill",
                  "knn", "mice", "drop_rows", "drop_columns"], index=0,
    help="'auto' picks a technique automatically based on how much data is missing.",
)

sb_label("Outliers")
remove_outliers = st.sidebar.toggle("Remove outliers", value=True)
outlier_method = st.sidebar.selectbox(
    "Method", ["auto", "iqr", "zscore", "isolation_forest", "lof", "dbscan"], index=0,
    disabled=not remove_outliers,
)

sb_label("Encoding & Scaling")
do_encode = st.sidebar.toggle("Encode categorical columns", value=True)
encoding_method = st.sidebar.selectbox(
    "Encoding method", ["auto", "label", "onehot", "ordinal", "frequency", "target"], index=0,
    disabled=not do_encode,
    help="'target' requires a target column above; 'auto' picks based on cardinality.",
)
do_scale = st.sidebar.toggle("Scale numeric features", value=True)
scaling_method = st.sidebar.selectbox(
    "Scaling method", ["auto", "minmax", "standard", "robust", "maxabs", "normalizer"], index=0,
    disabled=not do_scale,
)

sb_label("Extras")
run_feature_engineering = st.sidebar.toggle(
    "Generate feature-engineered dataset", value=False,
    help="Adds datetime part columns and text-length features; saved as a separate CSV.",
)
run_model_comparison = st.sidebar.toggle(
    "Compare multiple ML models", value=True,
    help="Requires a target column above. Trains several classifiers/regressors and ranks them.",
    disabled=not advanced_mode,
)
run_shap = st.sidebar.toggle(
    "Generate SHAP summary plot", value=False,
    help="Explains the best model's predictions. Requires the optional 'shap' package "
         "and adds some processing time; skipped automatically if 'shap' isn't installed.",
    disabled=not (advanced_mode and run_model_comparison),
)
build_profile = st.sidebar.toggle(
    "Generate full profiling report (slower)",
    value=True,
    help=f"Automatically skipped for datasets over {PROFILING_ROW_LIMIT:,} rows to avoid running out of memory on small deployments.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Supported formats: " + ", ".join(ext.strip(".").upper() for ext in SUPPORTED_EXTENSIONS)
)

# ----------------------------------------------------------------------
# Header / uploader
# ----------------------------------------------------------------------
st.caption("Upload any dataset and get automated cleaning, EDA, visualizations, and a full report.")

uploaded_file = st.file_uploader(
    "Drag and drop a dataset here, or click to browse",
    type=[ext.strip(".") for ext in SUPPORTED_EXTENSIONS],
)

# ----------------------------------------------------------------------
# Run pipeline on upload (or when pipeline options change)
# ----------------------------------------------------------------------
if uploaded_file is not None:
    options_signature = (
        advanced_mode, target_column_input.strip(), missing_technique, remove_outliers,
        outlier_method, do_encode, encoding_method, do_scale, scaling_method,
        run_feature_engineering, run_model_comparison, run_shap, build_profile,
    )
    run_key = f"{uploaded_file.name}_{uploaded_file.size}_{hash(options_signature)}"

    if st.session_state.get("last_run_key") != run_key:
        with st.spinner("Running AutoEDA Pro pipeline — loading, cleaning, analyzing, visualizing..."):
            work_dir = tempfile.mkdtemp(prefix="autoeda_")
            input_dir = os.path.join(work_dir, "input")
            output_dir = os.path.join(work_dir, "output")
            charts_dir = os.path.join(work_dir, "charts")
            os.makedirs(input_dir, exist_ok=True)

            saved_input_path = os.path.join(input_dir, uploaded_file.name)
            with open(saved_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Peek at row count/columns so we can auto-disable the (memory-heavy)
            # profiling report on large datasets, and validate the chosen
            # target column, regardless of the sidebar checkboxes.
            effective_build_profile = build_profile
            row_count = None
            target_column = target_column_input.strip() or None
            try:
                peek_df = load_dataset(saved_input_path)
                row_count = len(peek_df)
                if target_column and target_column not in peek_df.columns:
                    st.session_state["target_column_warning"] = (
                        f"Column '{target_column}' was not found in this dataset — "
                        f"ignoring it for this run."
                    )
                    target_column = None
                else:
                    st.session_state["target_column_warning"] = None
                del peek_df
            except Exception:
                pass

            profiling_skip_warning = None
            if build_profile and row_count is not None and row_count > PROFILING_ROW_LIMIT:
                effective_build_profile = False
                profiling_skip_warning = (
                    f"This dataset has {row_count:,} rows, over the {PROFILING_ROW_LIMIT:,}-row "
                    f"limit for the full profiling report — skipped it to avoid running out of "
                    f"memory. The cleaned CSV, charts, dashboard, and insights are unaffected."
                )
            st.session_state["profiling_skip_warning"] = profiling_skip_warning

            try:
                results = run_pipeline(
                    input_path=saved_input_path,
                    output_dir=output_dir,
                    charts_dir=charts_dir,
                    encode=do_encode,
                    scale=do_scale,
                    encoding_method=encoding_method,
                    scaling_method=scaling_method,
                    remove_outliers=remove_outliers,
                    build_profiling_report=effective_build_profile,
                    advanced=advanced_mode,
                    target_column=target_column,
                    missing_technique=missing_technique,
                    outlier_method=outlier_method,
                    run_feature_engineering=run_feature_engineering,
                    run_model_comparison=run_model_comparison,
                    run_shap=run_shap,
                )
                st.session_state["results"] = results
                st.session_state["work_dir"] = work_dir
                st.session_state["last_run_key"] = run_key
                st.session_state["error"] = None
            except Exception as e:
                st.session_state["error"] = str(e)

    error = st.session_state.get("error")
    if error:
        st.error(f"Something went wrong processing this file: {error}")
    else:
        results = st.session_state["results"]

        # -- Read supporting data back in for display -----------------
        cleaned_df = pd.read_csv(results["cleaned_csv"])
        with open(results["insights"], "r", encoding="utf-8") as f:
            insights_text = f.read()

        st.success(f"Done! Cleaned dataset: {cleaned_df.shape[0]} rows x {cleaned_df.shape[1]} columns.")

        if st.session_state.get("target_column_warning"):
            st.warning(st.session_state["target_column_warning"])
        if st.session_state.get("profiling_skip_warning"):
            st.warning(st.session_state["profiling_skip_warning"])

        quality_report = results.get("quality_report")

        # -- Tabs for detailed views -------------------------------------
        has_model_comparison = results.get("model_comparison") is not None and not results["model_comparison"].empty
        tab_labels = [
            "📈 Charts", "🧠 Insights & Recommendations", "📋 Cleaned Data",
            "🧪 Feature Lab", "🖥️ Full Dashboard", "🤖 ML Lab",
        ]
        if has_model_comparison:
            tab_labels.append("🏆 Model Comparison")
        tabs = st.tabs(tab_labels)
        tab_charts, tab_insights, tab_cleaned, tab_feature, tab_dashboard, tab_mllab = tabs[:6]
        tab_models = tabs[6] if has_model_comparison else None

        # ---- Charts -------------------------------------------------
        with tab_charts:
            with st.container(border=True):
                col_charts, col_stats = st.columns([3, 1])
                with col_charts:
                    st.markdown("#### 📈 Charts")
                    chart_paths = [p for p in results["charts"] if p and os.path.exists(p)]
                    cc = st.columns(2)
                    for i, path in enumerate(chart_paths):
                        cc[i % 2].image(
                            path,
                            caption=os.path.splitext(os.path.basename(path))[0].replace("_", " ").title(),
                            use_container_width=True,
                        )
                with col_stats:
                    mem_kb = cleaned_df.memory_usage(deep=True).sum() / 1024
                    stat_items = [
                        {"value": f"{cleaned_df.shape[0]:,}", "label": "Rows"},
                        {"value": f"{cleaned_df.shape[1]}", "label": "Columns"},
                        {"value": f"{mem_kb:,.2f} KB", "label": "Memory Usage"},
                    ]
                    if quality_report:
                        stat_items.append({
                            "value": f"{quality_report['quality_score']}/100",
                            "label": "Data Quality",
                            "badge": quality_report["quality_grade"],
                        })
                    else:
                        stat_items.append({
                            "value": int(cleaned_df.isnull().sum().sum()),
                            "label": "Missing values",
                        })
                    render_stat_cards(stat_items)

        # ---- Insights & Recommendations ------------------------------
        with tab_insights:
            with st.container(border=True):
                copy_col, _ = st.columns([9, 1])
                with copy_col:
                    st.markdown("#### 💡 Insights & Recommendations")
                insight_items, rec_items = parse_insights_file(insights_text)
                ic, rc = st.columns(2)
                with ic:
                    render_numbered_list("INSIGHTS", insight_items)
                with rc:
                    render_numbered_list("RECOMMENDATIONS", rec_items)
                with st.expander("Raw insights.txt"):
                    st.text(insights_text)

        # ---- Cleaned Data ---------------------------------------------
        with tab_cleaned:
            with st.container(border=True):
                st.markdown("#### 📋 Cleaned Data")
                st.dataframe(cleaned_df, use_container_width=True, height=520)

        # ---- Feature Lab -------------------------------------------
        with tab_feature:
            with st.container(border=True):
                st.markdown("#### 🧪 Feature Lab")
                fe_path = results.get("feature_engineered_csv")
                if fe_path and os.path.exists(fe_path):
                    st.caption(
                        "Generated feature-engineered dataset with datetime expansions "
                        "and text-length signal columns."
                    )
                    fe_df = pd.read_csv(fe_path)
                    st.dataframe(fe_df, use_container_width=True, height=420)
                    with open(fe_path, "rb") as f:
                        st.download_button(
                            "⬇ Download engineered CSV", f,
                            file_name=os.path.basename(fe_path), key="dl_fe_csv",
                        )
                else:
                    st.info(
                        "Enable **Generate feature-engineered dataset** in the sidebar "
                        "(under Extras) and re-run to populate this tab."
                    )

        # ---- Full Dashboard ---------------------------------------
        with tab_dashboard:
            with st.container(border=True):
                st.markdown("#### 🖥️ Full Dashboard")
                with open(results["dashboard"], "r", encoding="utf-8") as f:
                    dashboard_html = f.read()
                components.html(dashboard_html, height=1000, scrolling=True)

        # ---- Model Comparison -----------------------------------------
        if has_model_comparison:
            with tab_models:
                with st.container(border=True):
                    best_model_name = results.get("best_model_name")
                    st.markdown("#### 🏆 Model Comparison")
                    st.caption(
                        f"Task type: **{results.get('model_task_type')}** — models trained on the cleaned, "
                        f"preprocessed dataset and ranked by holdout performance."
                        + (f" Best model: **{best_model_name}**." if best_model_name else "")
                    )
                    st.dataframe(results["model_comparison"], use_container_width=True)

                    fi_path = results.get("feature_importance_csv")
                    fi_chart = results.get("feature_importance_chart")
                    if fi_path and os.path.exists(fi_path):
                        st.markdown("**Feature Importance (composite ranking)**")
                        if fi_chart and os.path.exists(fi_chart):
                            st.image(fi_chart, use_container_width=True)
                        st.dataframe(pd.read_csv(fi_path), use_container_width=True)

                    eval_path = results.get("evaluation_metrics_csv")
                    if eval_path and os.path.exists(eval_path):
                        st.markdown(f"**Detailed Evaluation Metrics — {best_model_name}**")
                        st.dataframe(pd.read_csv(eval_path), use_container_width=True)

                    cm_chart, roc_chart, resid_chart, shap_chart = (
                        results.get("confusion_matrix_chart"), results.get("roc_curve_chart"),
                        results.get("residual_plot_chart"), results.get("shap_summary_chart"),
                    )
                    if cm_chart or roc_chart:
                        st.markdown(f"**Classification Diagnostics — {best_model_name}**")
                        diag_cols = st.columns(2)
                        if cm_chart and os.path.exists(cm_chart):
                            diag_cols[0].image(cm_chart, caption="Confusion Matrix", use_container_width=True)
                        if roc_chart and os.path.exists(roc_chart):
                            diag_cols[1].image(roc_chart, caption="ROC Curve", use_container_width=True)
                    if resid_chart and os.path.exists(resid_chart):
                        st.markdown(f"**Residual Analysis — {best_model_name}**")
                        st.image(resid_chart, use_container_width=True)
                    if shap_chart and os.path.exists(shap_chart):
                        st.markdown("**SHAP Value Plot**")
                        st.image(shap_chart, use_container_width=True)
                    elif run_shap:
                        st.caption("SHAP plot unavailable — install the optional `shap` package to enable it.")

        # ---- ML Lab -----------------------------------------------------
        with tab_mllab:
            with st.container(border=True):
                st.markdown("#### 🤖 ML Lab")
                st.caption("Train a real scikit-learn model on your cleaned data — nothing leaves this session.")

                ml_col1, ml_col2 = st.columns(2)

                with ml_col1:
                    target_column = st.selectbox("Target column (what to predict)", cleaned_df.columns.tolist())
                    suggested = suggest_task_type(cleaned_df[target_column]) if target_column else "classification"
                    task_type = st.radio(
                        "Task type", ["classification", "regression"],
                        index=0 if suggested == "classification" else 1,
                        horizontal=True,
                        help=f"Auto-detected as '{suggested}' based on the target column — override if needed.",
                    )

                model_options = list(CLASSIFICATION_MODELS.keys()) if task_type == "classification" else list(REGRESSION_MODELS.keys())

                with ml_col2:
                    algorithm = st.selectbox("Algorithm", model_options)
                    test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

                train_clicked = st.button("▶ Train model", type="primary")

                if train_clicked:
                    with st.spinner("Training model..."):
                        try:
                            ml_result = train_and_evaluate(
                                cleaned_df, target_column=target_column,
                                task_type=task_type, algorithm=algorithm, test_size=test_size,
                            )
                            st.session_state["ml_result"] = ml_result
                            st.session_state["ml_error"] = None
                        except Exception as e:
                            st.session_state["ml_error"] = str(e)
                            st.session_state["ml_result"] = None

                ml_error = st.session_state.get("ml_error")
                ml_result = st.session_state.get("ml_result")

                if ml_error:
                    st.error(ml_error)

                if ml_result:
                    st.markdown(
                        f"**{ml_result['algorithm']}** trained on {ml_result['n_train']} rows, "
                        f"tested on {ml_result['n_test']} rows."
                    )

                    render_stat_cards(
                        [{"value": f"{v:.3f}", "label": k.upper()} for k, v in ml_result["metrics"].items()],
                        horizontal=True,
                    )
                    st.write("")

                    res_col1, res_col2 = st.columns(2)

                    with res_col1:
                        if ml_result["task_type"] == "classification":
                            st.markdown("**Confusion matrix**")
                            import matplotlib.pyplot as plt
                            import seaborn as sns

                            cm = ml_result["confusion_matrix"]
                            labels = ml_result.get("class_labels")
                            fig, ax = plt.subplots(figsize=(4.5, 4))
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                                        xticklabels=labels, yticklabels=labels, cbar=False)
                            ax.set_xlabel("Predicted")
                            ax.set_ylabel("Actual")
                            st.pyplot(fig)
                        else:
                            st.markdown("**Actual vs. Predicted**")
                            import matplotlib.pyplot as plt

                            pva = ml_result["predictions_vs_actual"]
                            fig, ax = plt.subplots(figsize=(4.5, 4))
                            ax.scatter(pva["actual"], pva["predicted"], alpha=0.6, color=palette["accent"])
                            lims = [min(pva["actual"] + pva["predicted"]), max(pva["actual"] + pva["predicted"])]
                            ax.plot(lims, lims, "--", color="gray")
                            ax.set_xlabel("Actual")
                            ax.set_ylabel("Predicted")
                            st.pyplot(fig)

                    with res_col2:
                        if ml_result["feature_importance"]:
                            st.markdown("**Feature importance**")
                            import matplotlib.pyplot as plt

                            fi = ml_result["feature_importance"]
                            fi_df = pd.DataFrame(
                                {"feature": list(fi.keys()), "importance": list(fi.values())}
                            ).head(10)
                            fi_sorted = fi_df.sort_values("importance", ascending=True)
                            bar_colors = [palette["accent"]] * len(fi_sorted)
                            if bar_colors:
                                bar_colors[-1] = palette["danger"]
                            fig, ax = plt.subplots(figsize=(4.5, 4))
                            ax.barh(fi_sorted["feature"], fi_sorted["importance"], color=bar_colors)
                            ax.set_xlabel("Importance")
                            fig.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.caption("This algorithm doesn't expose feature importances or coefficients.")

                    st.markdown("**Code that ran**")
                    st.code(ml_result["generated_code"], language="python")

                    # -- Model & bundle downloads -------------------------------
                    st.markdown("**Download**")
                    model_bytes = pickle.dumps({
                        "model": ml_result["model"],
                        "label_encoder": ml_result.get("label_encoder"),
                        "task_type": ml_result["task_type"],
                        "algorithm": ml_result["algorithm"],
                        "target_column": ml_result["target_column"],
                        "feature_names": ml_result["feature_names"],
                    })
                    model_filename = f"{ml_result['algorithm']}_model.pkl"

                    # Prefer the profiling report as "the report" in the bundle;
                    # fall back to the dashboard HTML if profiling was skipped.
                    report_path = results.get("profiling_report") or results["dashboard"]

                    dl_col1, dl_col2 = st.columns(2)
                    dl_col1.download_button(
                        "🧠 Download trained model (.pkl)",
                        model_bytes,
                        file_name=model_filename,
                        key="dl_model_pkl",
                    )

                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        zf.write(results["cleaned_csv"], arcname=os.path.basename(results["cleaned_csv"]))
                        zf.write(report_path, arcname=os.path.basename(report_path))
                        zf.writestr(model_filename, model_bytes)
                    zip_buffer.seek(0)

                    dl_col2.download_button(
                        "📦 Download bundle (CSV + Report + Model .zip)",
                        zip_buffer,
                        file_name="autoeda_pro_bundle.zip",
                        mime="application/zip",
                        key="dl_bundle_zip",
                    )

else:
    st.info("👆 Upload a dataset to get started. A sample file is available in the `input/` folder if you want to try the app first.")
