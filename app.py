"""
app.py
------
Streamlit web interface for AutoEDA Pro.

Lets the user drag-and-drop (or browse for) a dataset in any supported
format — CSV, TSV, TXT, XLSX, XLS, JSON, or Parquet — and runs the full
AutoEDA Pro pipeline on it, then displays the results (metrics, charts,
insights, recommendations, dashboard) right in the browser with download
buttons for every generated file.

Run with:
    streamlit run app.py
"""

import os
import shutil
import tempfile

import streamlit as st
import streamlit.components.v1 as components

from main import run_pipeline
from utils import SUPPORTED_EXTENSIONS
from model_training import train_and_evaluate, CLASSIFICATION_MODELS, REGRESSION_MODELS, suggest_task_type

st.set_page_config(page_title="AutoEDA Pro", page_icon="📊", layout="wide")

# Above this many rows, the full profiling report is skipped automatically
# (even if requested) to stay within the ~1 GB memory budget of small
# deployments such as Streamlit Community Cloud's free tier.
PROFILING_ROW_LIMIT = 50_000

# ----------------------------------------------------------------------
# Sidebar - options
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Pipeline Options")
encoding_method = st.sidebar.selectbox("Categorical encoding", ["label", "onehot"], index=0)
scaling_method = st.sidebar.selectbox("Feature scaling", ["minmax", "standard"], index=0)
do_encode = st.sidebar.checkbox("Encode categorical columns", value=True)
do_scale = st.sidebar.checkbox("Scale numeric features", value=True)
remove_outliers = st.sidebar.checkbox("Remove outliers (IQR method)", value=True)
build_profile = st.sidebar.checkbox(
    "Generate full profiling report (slower)",
    value=True,
    help=f"Automatically skipped for datasets over {PROFILING_ROW_LIMIT:,} rows to avoid running out of memory on small deployments.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Supported formats: " + ", ".join(ext.strip(".").upper() for ext in SUPPORTED_EXTENSIONS)
)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("📊 AutoEDA Pro")
st.caption("Upload any dataset and get automated cleaning, EDA, visualizations, and a full report.")

uploaded_file = st.file_uploader(
    "Drag and drop a dataset here, or click to browse",
    type=[ext.strip(".") for ext in SUPPORTED_EXTENSIONS],
)

# ----------------------------------------------------------------------
# Run pipeline on upload
# ----------------------------------------------------------------------
if uploaded_file is not None:
    run_key = f"{uploaded_file.name}_{uploaded_file.size}"

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

            # Peek at row count so we can auto-disable the (memory-heavy)
            # profiling report on large datasets, regardless of the
            # sidebar checkbox.
            from utils import load_dataset as _peek_load

            effective_build_profile = build_profile
            row_count = None
            try:
                peek_df = _peek_load(saved_input_path)
                row_count = len(peek_df)
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
        import pandas as pd

        cleaned_df = pd.read_csv(results["cleaned_csv"])
        with open(results["insights"], "r", encoding="utf-8") as f:
            insights_text = f.read()

        st.success(f"Done! Cleaned dataset: {cleaned_df.shape[0]} rows x {cleaned_df.shape[1]} columns.")

        if st.session_state.get("profiling_skip_warning"):
            st.warning(st.session_state["profiling_skip_warning"])

        # -- Top-level metrics -----------------------------------------
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows (cleaned)", cleaned_df.shape[0])
        col2.metric("Columns (cleaned)", cleaned_df.shape[1])
        col3.metric("Missing values", int(cleaned_df.isnull().sum().sum()))
        col4.metric("Duplicate rows", int(cleaned_df.duplicated().sum()))

        # -- Downloads ---------------------------------------------------
        st.subheader("⬇️ Downloads")
        d1, d2, d3, d4 = st.columns(4)
        with open(results["cleaned_csv"], "rb") as f:
            d1.download_button("Cleaned CSV", f, file_name=os.path.basename(results["cleaned_csv"]))
        with open(results["dashboard"], "rb") as f:
            d2.download_button("Dashboard HTML", f, file_name="dashboard.html")
        if results.get("profiling_report"):
            with open(results["profiling_report"], "rb") as f:
                d3.download_button("Profiling Report", f, file_name="profiling_report.html")
        with open(results["insights"], "rb") as f:
            d4.download_button("Insights (.txt)", f, file_name="insights.txt")

        # -- Tabs for detailed views -------------------------------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📈 Charts", "🧠 Insights & Recommendations", "📋 Cleaned Data", "🖥️ Full Dashboard", "🤖 ML Lab"]
        )

        with tab1:
            chart_paths = [p for p in results["charts"] if p and os.path.exists(p)]
            cols = st.columns(2)
            for i, path in enumerate(chart_paths):
                cols[i % 2].image(path, caption=os.path.splitext(os.path.basename(path))[0].title(), use_container_width=True)

        with tab2:
            st.text(insights_text)

        with tab3:
            st.dataframe(cleaned_df, use_container_width=True)

        with tab4:
            with open(results["dashboard"], "r", encoding="utf-8") as f:
                dashboard_html = f.read()
            components.html(dashboard_html, height=1000, scrolling=True)

        with tab5:
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
                st.markdown(f"**{ml_result['algorithm']}** trained on {ml_result['n_train']} rows, tested on {ml_result['n_test']} rows.")

                metric_cols = st.columns(len(ml_result["metrics"]))
                for i, (k, v) in enumerate(ml_result["metrics"].items()):
                    metric_cols[i].metric(k.upper(), f"{v:.3f}")

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
                        ax.scatter(pva["actual"], pva["predicted"], alpha=0.6, color="#2b50e0")
                        lims = [min(pva["actual"] + pva["predicted"]), max(pva["actual"] + pva["predicted"])]
                        ax.plot(lims, lims, "--", color="gray")
                        ax.set_xlabel("Actual")
                        ax.set_ylabel("Predicted")
                        st.pyplot(fig)

                with res_col2:
                    if ml_result["feature_importance"]:
                        st.markdown("**Feature importance**")
                        fi = ml_result["feature_importance"]
                        fi_df = pd.DataFrame({"feature": list(fi.keys()), "importance": list(fi.values())}).head(10)
                        st.bar_chart(fi_df.set_index("feature"))
                    else:
                        st.caption("This algorithm doesn't expose feature importances or coefficients.")

                st.markdown("**Code that ran**")
                st.code(ml_result["generated_code"], language="python")

else:
    st.info("👆 Upload a dataset to get started. A sample file is available in the `input/` folder if you want to try the app first.")
