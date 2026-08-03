"""
main.py
-------
Entry point for AutoEDA Pro. Runs the complete automated workflow:

  Load Dataset -> Dataset Information -> Missing Value Analysis ->
  Duplicate Detection -> Data Preprocessing -> Outlier Detection ->
  Encoding -> Feature Scaling -> Visualization -> Correlation Analysis ->
  Automatic Insights -> HTML Dashboard -> Profiling Report -> Final Output

Usage:
    python main.py --input input/customers.csv
    python main.py --input input/customers.csv --no-scale --no-encode
"""

import argparse
import os
import sys
import pandas as pd

from utils import load_dataset, ensure_dirs
from eda import EDAAnalyzer
from preprocessing import DataPreprocessor
from visualization import ChartGenerator
from report_generator import (
    generate_insights,
    generate_recommendations,
    save_insights_file,
    generate_profiling_report,
)
from dashboard import build_dashboard


def run_pipeline(input_path, output_dir="output", charts_dir="charts",
                  encode=True, scale=True, encoding_method="label",
                  scaling_method="minmax", remove_outliers=True,
                  build_profiling_report=True):

    ensure_dirs(output_dir, charts_dir)
    dataset_name = os.path.splitext(os.path.basename(input_path))[0]

    print(f"[1/9] Loading dataset from {input_path} ...")
    df = load_dataset(input_path)
    print(f"      Loaded {df.shape[0]} rows x {df.shape[1]} columns.")

    print("[2/9] Running EDA on raw dataset ...")
    raw_eda = EDAAnalyzer(df)
    overview = raw_eda.dataset_overview()
    missing_analysis = raw_eda.missing_value_analysis()
    duplicate_info = raw_eda.duplicate_analysis()

    print("[3/9] Preprocessing dataset (missing values, duplicates, outliers, cleaning) ...")
    preprocessor = DataPreprocessor(df)
    cleaned_df = preprocessor.run_full_pipeline(
        encode=encode,
        scale=scale,
        encoding_method=encoding_method,
        scaling_method=scaling_method,
        remove_outlier_rows=remove_outliers,
    )
    preprocess_log = preprocessor.get_log()
    print(f"      Duplicates removed: {preprocess_log['duplicates_removed']}")
    print(f"      Outliers removed:   {preprocess_log['outliers_removed']}")

    cleaned_csv_path = os.path.join(output_dir, f"{dataset_name}_cleaned.csv")
    cleaned_df.to_csv(cleaned_csv_path, index=False)
    print(f"[4/9] Cleaned dataset saved to {cleaned_csv_path}")

    print("[5/9] Running full EDA (statistics + correlation) on cleaned dataset ...")
    # Use a numeric-preserving copy of the cleaned data for statistics,
    # but the ORIGINAL (pre-scaling) preprocessed frame for readability.
    stats_source = preprocessor.original_df.copy()
    stats_eda = EDAAnalyzer(cleaned_df)
    stats_summary = stats_eda.statistical_summary()
    stats_df = pd.DataFrame(stats_summary).T
    correlation_results = stats_eda.correlation_analysis()

    print("[6/9] Generating visualizations ...")
    chart_gen = ChartGenerator(df, output_dir=charts_dir)  # chart on raw-ish data for interpretability
    chart_paths = chart_gen.generate_all()
    print(f"      {len(chart_paths)} charts generated in {charts_dir}/")

    print("[7/9] Generating automatic insights & recommendations ...")
    insights = generate_insights(df, cleaned_df, overview, correlation_results, preprocess_log)
    recommendations = generate_recommendations(cleaned_df, correlation_results, overview)
    insights_path = save_insights_file(insights, recommendations, path=os.path.join(output_dir, "insights.txt"))
    print(f"      Insights saved to {insights_path}")

    print("[8/9] Building HTML dashboard ...")
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
    )
    print(f"      Dashboard saved to {dashboard_path}")

    profiling_path = None
    if build_profiling_report:
        print("[9/9] Generating profiling report (this may take a moment) ...")
        profiling_path = generate_profiling_report(
            df, output_path=os.path.join(output_dir, "profiling_report.html"),
            title=f"AutoEDA Pro - {dataset_name} Profiling Report",
        )
        print(f"      Profiling report saved to {profiling_path}")
    else:
        print("[9/9] Skipped profiling report (--no-profiling flag set).")

    print("\nAutoEDA Pro run complete.")
    print(f"  Clean dataset:     {cleaned_csv_path}")
    print(f"  Dashboard:         {dashboard_path}")
    if profiling_path:
        print(f"  Profiling report:  {profiling_path}")
    print(f"  Insights:          {insights_path}")
    print(f"  Charts directory:  {charts_dir}/")

    return {
        "cleaned_csv": cleaned_csv_path,
        "dashboard": dashboard_path,
        "profiling_report": profiling_path,
        "insights": insights_path,
        "charts": chart_paths,
    }


def main():
    parser = argparse.ArgumentParser(description="AutoEDA Pro - Automated EDA & Preprocessing Dashboard")
    parser.add_argument("--input", default="input/customers.csv", help="Path to input CSV dataset")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--charts", default="charts", help="Charts directory")
    parser.add_argument("--no-encode", action="store_true", help="Skip categorical encoding")
    parser.add_argument("--no-scale", action="store_true", help="Skip feature scaling")
    parser.add_argument("--encoding-method", default="label", choices=["label", "onehot"])
    parser.add_argument("--scaling-method", default="minmax", choices=["minmax", "standard"])
    parser.add_argument("--no-outlier-removal", action="store_true", help="Skip IQR outlier removal")
    parser.add_argument("--no-profiling", action="store_true", help="Skip profiling report generation")
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
    )


if __name__ == "__main__":
    main()
