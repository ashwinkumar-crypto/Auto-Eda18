"""
main.py
-------
Entry point for AutoEDA Pro. Runs the complete automated workflow:

  Load Dataset -> Data Quality Report -> Dataset Information ->
  Advanced Preprocessing (missing values, duplicates, outliers,
  encoding, scaling) -> Feature Engineering -> Visualization ->
  Correlation Analysis -> Feature Selection -> Model Comparison ->
  Automatic Insights -> HTML Dashboard -> Profiling Report -> Final Output

Usage:
    python main.py --input input/customers.csv
    python main.py --input input/customers.csv --target Purchased
    python main.py --input input/customers.csv --simple --no-scale --no-encode
"""

import argparse
import logging
import os
import sys
import warnings

import pandas as pd

from sklearn.metrics import confusion_matrix as _sk_confusion_matrix, classification_report as _sk_classification_report

from utils import load_dataset, ensure_dirs
from eda import EDAAnalyzer
from preprocessing import DataPreprocessor
from visualization import ChartGenerator
from data_quality import DataQualityReport
from feature_engineering import FeatureEngineer
from feature_selection import FeatureSelector
from outliers import OutlierDetector
from report_generator import (
    generate_insights,
    generate_recommendations,
    save_insights_file,
    save_dataframe_csv,
    generate_profiling_report,
)
from dashboard import build_dashboard

try:
    from ml_advanced import (
        detect_task_type, compare_classification_models, compare_regression_models,
        get_classification_diagnostics, get_regression_diagnostics,
    )
    from ml_visuals import (
        plot_confusion_matrix, plot_roc_curve, plot_residuals,
        plot_feature_importance, plot_shap_summary,
    )
    _HAS_ML_ADVANCED = True
except Exception:  # pragma: no cover - defensive
    _HAS_ML_ADVANCED = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("autoeda.main")
warnings.filterwarnings("ignore", category=FutureWarning)


def run_pipeline(input_path, output_dir="output", charts_dir="charts",
                  encode=True, scale=True, encoding_method="label",
                  scaling_method="minmax", remove_outliers=True,
                  build_profiling_report=True,
                  # -- advanced options (opt-in, all default to sensible auto behavior) --
                  advanced=True, target_column=None,
                  missing_technique="auto", outlier_method="auto",
                  run_feature_engineering=False, run_model_comparison=True,
                  run_shap=False):
    """
    Run the full AutoEDA Pro pipeline.

    When `advanced=True` (default), preprocessing uses
    DataPreprocessor.run_advanced_pipeline() with auto-recommended (or
    explicitly chosen) techniques per stage, and the pipeline also
    produces a data quality report, feature importance ranking, and a
    multi-model comparison (if `target_column` is supplied and
    scikit-learn/xgboost/lightgbm models are available).

    When `advanced=False`, behavior matches the original AutoEDA Pro
    pipeline exactly (simple median/mode imputation, IQR outlier
    removal, single encoding/scaling method, no quality score / feature
    selection / model comparison).
    """

    ensure_dirs(output_dir, charts_dir)
    dataset_name = os.path.splitext(os.path.basename(input_path))[0]

    logger.info("[1/11] Loading dataset from %s ...", input_path)
    df = load_dataset(input_path)
    logger.info("Loaded %d rows x %d columns.", df.shape[0], df.shape[1])

    logger.info("[2/11] Running EDA on raw dataset ...")
    raw_eda = EDAAnalyzer(df)
    overview = raw_eda.dataset_overview()
    missing_analysis = raw_eda.missing_value_analysis()
    duplicate_info = raw_eda.duplicate_analysis()

    logger.info("[3/11] Generating data quality report ...")
    quality_report = None
    try:
        quality_report = DataQualityReport(df, target_column=target_column).generate()
        logger.info("Data quality score: %s (%s)", quality_report["quality_score"], quality_report["quality_grade"])
    except Exception:
        logger.exception("Data quality report generation failed; continuing without it.")

    logger.info("[4/11] Preprocessing dataset (missing values, duplicates, outliers, "
                "encoding, scaling) ...")
    preprocessor = DataPreprocessor(df)
    outlier_summary_df = None
    try:
        if advanced:
            # Preview outlier method comparison before committing to one (for reporting).
            try:
                outlier_summary_df = OutlierDetector(df).compare_methods(["iqr", "zscore", "isolation_forest"])
            except Exception:
                logger.exception("Outlier method comparison failed; continuing.")

            cleaned_df = preprocessor.run_advanced_pipeline(
                target_column=target_column,
                missing_technique=missing_technique,
                remove_outlier_rows=remove_outliers,
                outlier_method=outlier_method,
                encode=encode,
                encoding_method=encoding_method,
                scale=scale,
                scaling_method=scaling_method,
            )
        else:
            cleaned_df = preprocessor.run_full_pipeline(
                encode=encode, scale=scale, encoding_method=encoding_method,
                scaling_method=scaling_method, remove_outlier_rows=remove_outliers,
            )
    except Exception:
        logger.exception("Preprocessing failed.")
        raise
    preprocess_log = preprocessor.get_log()
    logger.info("Duplicates removed: %s", preprocess_log.get("duplicates_removed"))
    logger.info("Outliers removed: %s", preprocess_log.get("outliers_removed"))

    preprocessing_summary_df = preprocessor.get_preprocessing_summary() if advanced else None

    cleaned_csv_path = os.path.join(output_dir, f"{dataset_name}_cleaned.csv")
    cleaned_df.to_csv(cleaned_csv_path, index=False)
    logger.info("[5/11] Cleaned dataset saved to %s", cleaned_csv_path)

    # -- Optional feature engineering (opt-in; can expand dimensionality) --
    feature_engineered_path = None
    if run_feature_engineering:
        logger.info("Running feature engineering (datetime expansion, text length) ...")
        try:
            fe = FeatureEngineer(preprocessor.original_df)
            fe_df = fe.run_all()
            feature_engineered_path = os.path.join(output_dir, f"{dataset_name}_feature_engineered.csv")
            fe_df.to_csv(feature_engineered_path, index=False)
            logger.info("Feature-engineered dataset saved to %s", feature_engineered_path)
        except Exception:
            logger.exception("Feature engineering failed; continuing without it.")

    logger.info("[6/11] Running full EDA (statistics + correlation) on cleaned dataset ...")
    stats_eda = EDAAnalyzer(cleaned_df)
    stats_summary = stats_eda.statistical_summary()
    stats_df = pd.DataFrame(stats_summary).T
    correlation_results = stats_eda.correlation_analysis()

    logger.info("[7/11] Generating visualizations ...")
    chart_gen = ChartGenerator(df, output_dir=charts_dir)  # chart on raw-ish data for interpretability
    chart_paths = chart_gen.generate_all(include_extended=advanced)
    logger.info("%d charts generated in %s/", len(chart_paths), charts_dir)

    # -- Feature selection / importance (requires a target column) --
    feature_importance_df = None
    feature_importance_path = None
    if advanced and target_column and target_column in cleaned_df.columns:
        logger.info("[8/11] Running feature selection / importance ranking ...")
        try:
            selector = FeatureSelector(cleaned_df, target_column=target_column)
            feature_importance_df = selector.ranked_report()
            feature_importance_path = save_dataframe_csv(
                feature_importance_df, os.path.join(output_dir, "feature_importance.csv"))
        except Exception:
            logger.exception("Feature selection failed; continuing without it.")
    else:
        logger.info("[8/11] Skipped feature selection (no target column specified).")

    # -- Model comparison (requires a target column and ml_advanced availability) --
    model_comparison_df = None
    model_metrics_path = None
    model_comparison_path = None
    evaluation_metrics_path = None
    model_task_type = None
    best_model_name = None
    confusion_matrix_path = None
    roc_curve_path = None
    residual_plot_path = None
    feature_importance_chart_path = None
    shap_summary_path = None
    classification_report_text = None

    if advanced and run_model_comparison and _HAS_ML_ADVANCED and target_column and target_column in cleaned_df.columns:
        logger.info("[9/11] Training and comparing candidate ML models ...")
        try:
            model_task_type = detect_task_type(cleaned_df, target_column)
            if model_task_type == "classification":
                model_comparison_df = compare_classification_models(cleaned_df, target_column)
            elif model_task_type == "regression":
                model_comparison_df = compare_regression_models(cleaned_df, target_column)

            model_metrics_path = save_dataframe_csv(
                model_comparison_df, os.path.join(output_dir, "model_metrics.csv"))
            model_comparison_path = save_dataframe_csv(
                model_comparison_df, os.path.join(output_dir, "model_comparison.csv"))

            if model_comparison_df is not None and not model_comparison_df.empty:
                best_model_name = model_comparison_df.iloc[0]["model"]
                logger.info("Best model: %s", best_model_name)

                # -- Detailed diagnostics + plots for the best model --
                try:
                    if model_task_type == "classification":
                        diag = get_classification_diagnostics(cleaned_df, target_column, best_model_name)
                        evaluation_metrics_path = save_dataframe_csv(
                            pd.DataFrame([diag["evaluation_metrics"]]),
                            os.path.join(output_dir, "evaluation_metrics.csv"))
                        confusion_matrix_path = plot_confusion_matrix(
                            _sk_confusion_matrix(diag["y_test"], diag["y_pred"]),
                            diag["labels"], output_dir=charts_dir)
                        roc_curve_path = plot_roc_curve(diag["y_test"], diag["y_proba"], output_dir=charts_dir)
                        classification_report_text = _sk_classification_report(
                            diag["y_test"], diag["y_pred"], target_names=diag["labels"], zero_division=0)
                        if run_shap:
                            shap_summary_path = plot_shap_summary(diag["model"], diag["X_test"], output_dir=charts_dir)
                    elif model_task_type == "regression":
                        diag = get_regression_diagnostics(cleaned_df, target_column, best_model_name)
                        evaluation_metrics_path = save_dataframe_csv(
                            pd.DataFrame([diag["evaluation_metrics"]]),
                            os.path.join(output_dir, "evaluation_metrics.csv"))
                        residual_plot_path = plot_residuals(diag["y_test"], diag["y_pred"], output_dir=charts_dir)
                        if run_shap:
                            shap_summary_path = plot_shap_summary(diag["model"], diag["X_test"], output_dir=charts_dir)
                except Exception:
                    logger.exception("Best-model diagnostics/plots failed; continuing without them.")
        except Exception:
            logger.exception("Model comparison failed; continuing without it.")
    else:
        logger.info("[9/11] Skipped model comparison (no target column, or feature disabled).")

    # -- Feature importance bar chart (independent of model comparison) --
    if feature_importance_df is not None and not feature_importance_df.empty:
        try:
            feature_importance_chart_path = plot_feature_importance(feature_importance_df, output_dir=charts_dir)
        except Exception:
            logger.exception("Feature importance chart generation failed; continuing without it.")

    preprocessing_summary_path = save_dataframe_csv(
        preprocessing_summary_df, os.path.join(output_dir, "preprocessing_summary.csv"))

    logger.info("[10/11] Generating automatic insights & recommendations ...")
    insights = generate_insights(
        df, cleaned_df, overview, correlation_results, preprocess_log,
        quality_report=quality_report, feature_importance_df=feature_importance_df,
        model_comparison_df=model_comparison_df,
    )
    recommendations = generate_recommendations(
        cleaned_df, correlation_results, overview,
        quality_report=quality_report, model_comparison_df=model_comparison_df,
        task_type=model_task_type,
    )
    insights_path = save_insights_file(insights, recommendations, path=os.path.join(output_dir, "insights.txt"))
    logger.info("Insights saved to %s", insights_path)

    logger.info("[11/11] Building HTML dashboard ...")
    conclusion = (
        f"The dataset '{dataset_name}' was automatically cleaned, analyzed, and visualized. "
        f"After preprocessing, it contains {cleaned_df.shape[0]} rows and {cleaned_df.shape[1]} columns "
        f"with no missing values and duplicates removed, making it ready for machine learning workflows."
    )
    dashboard_path = build_dashboard(
        dataset_name=dataset_name,
        overview=overview,
        stats_df=stats_df,
        missing_df=missing_analysis,
        duplicate_info=duplicate_info,
        chart_paths=chart_paths,
        insights=insights,
        recommendations=recommendations,
        conclusion=conclusion,
        output_path=os.path.join(output_dir, "dashboard.html"),
        quality_report=quality_report,
        preprocessing_summary_df=preprocessing_summary_df,
        outlier_summary_df=outlier_summary_df,
        feature_importance_df=feature_importance_df,
        model_comparison_df=model_comparison_df,
        model_task_type=model_task_type,
        confusion_matrix_path=confusion_matrix_path,
        roc_curve_path=roc_curve_path,
        residual_plot_path=residual_plot_path,
        feature_importance_chart_path=feature_importance_chart_path,
        shap_summary_path=shap_summary_path,
        classification_report_text=classification_report_text,
        best_model_name=best_model_name,
    )
    logger.info("Dashboard saved to %s", dashboard_path)

    profiling_path = None
    if build_profiling_report:
        logger.info("Generating profiling report (this may take a moment) ...")
        profiling_path = generate_profiling_report(
            df, output_path=os.path.join(output_dir, "profiling_report.html"),
            title=f"AutoEDA Pro - {dataset_name} Profiling Report",
        )
        logger.info("Profiling report saved to %s", profiling_path)
    else:
        logger.info("Skipped profiling report (--no-profiling flag set).")

    logger.info("AutoEDA Pro run complete.")

    return {
        "cleaned_csv": cleaned_csv_path,
        "feature_engineered_csv": feature_engineered_path,
        "dashboard": dashboard_path,
        "profiling_report": profiling_path,
        "insights": insights_path,
        "charts": chart_paths,
        "quality_report": quality_report,
        "preprocessing_summary_csv": preprocessing_summary_path,
        "feature_importance_csv": feature_importance_path,
        "model_metrics_csv": model_metrics_path,
        "model_comparison_csv": model_comparison_path,
        "evaluation_metrics_csv": evaluation_metrics_path,
        "model_comparison": model_comparison_df,
        "model_task_type": model_task_type,
        "best_model_name": best_model_name,
        "confusion_matrix_chart": confusion_matrix_path,
        "roc_curve_chart": roc_curve_path,
        "residual_plot_chart": residual_plot_path,
        "feature_importance_chart": feature_importance_chart_path,
        "shap_summary_chart": shap_summary_path,
    }


def main():
    parser = argparse.ArgumentParser(description="AutoEDA Pro - Automated EDA & Preprocessing Dashboard")
    parser.add_argument("--input", default="input/customers.csv", help="Path to input dataset (CSV/TSV/XLSX/JSON/Parquet)")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--charts", default="charts", help="Charts directory")
    parser.add_argument("--target", default=None, help="Target column for quality/feature-selection/model comparison")
    parser.add_argument("--simple", action="store_true",
                         help="Use the original simple pipeline (median/mode imputation, IQR outliers only, "
                              "no quality score / feature selection / model comparison)")
    parser.add_argument("--no-encode", action="store_true", help="Skip categorical encoding")
    parser.add_argument("--no-scale", action="store_true", help="Skip feature scaling")
    parser.add_argument("--encoding-method", default="label",
                         choices=["auto", "label", "onehot", "ordinal", "frequency", "target"])
    parser.add_argument("--scaling-method", default="minmax",
                         choices=["auto", "minmax", "standard", "robust", "maxabs", "normalizer"])
    parser.add_argument("--missing-technique", default="auto",
                         choices=["auto", "drop_rows", "drop_columns", "mean", "median", "mode",
                                  "constant", "ffill", "bfill", "knn", "mice"])
    parser.add_argument("--outlier-method", default="auto",
                         choices=["auto", "iqr", "zscore", "isolation_forest", "lof", "dbscan"])
    parser.add_argument("--no-outlier-removal", action="store_true", help="Skip outlier removal")
    parser.add_argument("--no-profiling", action="store_true", help="Skip profiling report generation")
    parser.add_argument("--no-model-comparison", action="store_true", help="Skip multi-model ML comparison")
    parser.add_argument("--feature-engineering", action="store_true",
                         help="Also produce a feature-engineered dataset (datetime expansion, text length)")
    parser.add_argument("--shap", action="store_true",
                         help="Also generate a SHAP summary plot for the best model (requires the optional "
                              "'shap' package; slower, so off by default)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        charts_dir=args.charts,
        encode=not args.no_encode,
        scale=not args.no_scale,
        encoding_method=args.encoding_method,
        scaling_method=args.scaling_method,
        remove_outliers=not args.no_outlier_removal,
        build_profiling_report=not args.no_profiling,
        advanced=not args.simple,
        target_column=args.target,
        missing_technique=args.missing_technique,
        outlier_method=args.outlier_method,
        run_feature_engineering=args.feature_engineering,
        run_model_comparison=not args.no_model_comparison,
        run_shap=args.shap,
    )


if __name__ == "__main__":
    main()
