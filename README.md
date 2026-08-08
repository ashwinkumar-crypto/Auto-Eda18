# AutoEDA Pro

### Professional Automated Data Preprocessing & Exploratory Data Analysis Dashboard

AutoEDA Pro is a Python application that automates the complete Exploratory
Data Analysis (EDA) workflow — turning a raw CSV into a cleaned,
ML-ready dataset plus a set of professional reports, charts, and an
interactive HTML dashboard, in a single command.

## Deploying to Streamlit Community Cloud

The project is deploy-ready out of the box:

1. Push the repo to GitHub (must include `app.py`, all `.py` modules,
   `requirements.txt`, and `.streamlit/config.toml`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app**
   → point it at your repo/branch with main file path `app.py`.
3. Click **Deploy**.

Two things are already configured for small/free deployments:

* **`.streamlit/config.toml`** raises the upload limit to 500 MB and sets
  a matching light theme.
* **Automatic profiling guard:** datasets over 50,000 rows automatically
  skip the full profiling report (`fg-data-profiling` is memory-hungry),
  even if the sidebar checkbox is on — a warning explains why. The
  cleaned CSV, charts, dashboard, and insights are unaffected. Adjust
  `PROFILING_ROW_LIMIT` at the top of `app.py` if your deployment has
  more memory to work with.

## Quick Start — Web App (drag-and-drop upload)

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens a browser tab with a drag-and-drop upload box. Drop in **any**
CSV, TSV, TXT, XLSX, XLS, JSON, or Parquet file and the app will:

* run the full pipeline automatically,
* show live metrics, all 9 charts, insights/recommendations, and the
  cleaned data table right in the browser,
* embed the full interactive dashboard inline,
* give you one-click downloads for the cleaned CSV, `dashboard.html`,
  `profiling_report.html`, and `insights.txt`.

Sidebar options let you toggle encoding/scaling method, outlier removal,
and whether to build the (slower) full profiling report — all without
touching any code.

### 🤖 ML Lab

A new tab lets you go straight from cleaned data to a trained model:

* Pick any column as the **target** — the task type (classification vs.
  regression) is auto-detected from it, with a manual override
* Choose an **algorithm**: `RandomForest`, `LogisticRegression`,
  `KNeighbors`, `SVC` for classification; `RandomForest`,
  `LinearRegression`, `KNeighbors`, `SVR` for regression
* Adjust the **train/test split** with a slider
* Click **Train model** to see real metrics (accuracy/precision/recall/F1
  or R²/MAE/RMSE), a confusion matrix or actual-vs-predicted plot, feature
  importance, and the **exact scikit-learn code that ran** — so you can
  copy it straight into a notebook or script.

This runs a real `scikit-learn` model on your actual cleaned dataset (not
a simulation) — nothing leaves your machine/session.

## Quick Start — Command Line

```bash
pip install -r requirements.txt
python main.py --input input/customers.csv
```

Outputs are written to `output/` (cleaned CSV, dashboard, profiling
report, insights) and `charts/` (PNG visualizations).

## CLI Options

| Flag | Description | Default |
|---|---|---|
| `--input` | Path to input CSV | `input/customers.csv` |
| `--output` | Output directory | `output` |
| `--charts` | Charts directory | `charts` |
| `--no-encode` | Skip categorical encoding | encoding on |
| `--no-scale` | Skip feature scaling | scaling on |
| `--encoding-method` | `label` or `onehot` | `label` |
| `--scaling-method` | `minmax` or `standard` | `minmax` |
| `--no-outlier-removal` | Skip IQR outlier removal | removal on |
| `--no-profiling` | Skip the profiling report step | profiling on |

## Workflow

```
Load Dataset -> Dataset Information -> Missing Value Analysis ->
Duplicate Detection -> Data Preprocessing -> Outlier Detection ->
Encoding -> Feature Scaling -> Visualization -> Correlation Analysis ->
Automatic Insights -> HTML Dashboard -> Profiling Report -> Final Output
```

## Modules

* `app.py` — Streamlit web app: drag-and-drop upload for any supported
  format, runs the pipeline, and displays everything in the browser
* `utils.py` — multi-format dataset loading (CSV/TSV/TXT/XLSX/XLS/JSON/Parquet),
  encoding detection, formatting helpers
* `preprocessing.py` — `DataPreprocessor`: missing values, duplicates,
  IQR outlier removal, constant-column cleanup, encoding, scaling
* `eda.py` — `EDAAnalyzer`: overview, statistical summary, missing/duplicate
  analysis, Pearson/Spearman/Kendall correlation
* `visualization.py` — `ChartGenerator`: histogram, box plot, count plot,
  scatter plot, correlation heatmap, pair plot, bar chart, pie chart, line chart
* `report_generator.py` — automatic insights, recommendations, `insights.txt`,
  and the profiling report (via `fg-data-profiling`, with automatic fallback)
* `dashboard.py` — renders the self-contained `dashboard.html` (Jinja2,
  charts embedded as base64 so it works as a single file)
* `model_training.py` — ML Lab: trains a real scikit-learn model
  (classification or regression) on the cleaned data, with metrics,
  confusion matrix / actual-vs-predicted, feature importance, and a
  generated code snippet
* `main.py` — CLI entry point orchestrating the full pipeline

## Output Files

| File | Description |
|---|---|
| `output/<name>_cleaned.csv` | Cleaned, encoded, scaled dataset |
| `output/dashboard.html` | Self-contained interactive dashboard |
| `output/profiling_report.html` | Full variable-by-variable profiling report |
| `output/insights.txt` | Automatic insights & recommendations |
| `charts/*.png` | Individual chart images |

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, `fg-data-profiling`, Jinja2, chardet.

## Notes on the Sample Data

`input/customers.csv` is a synthetic sample (500+ rows) with intentionally
injected missing values, duplicate rows, and salary outliers so you can see
every stage of the pipeline (imputation, dedup, IQR removal) do real work.
Swap in your own CSV at any time — just point `--input` at it.

## Future Enhancements

Support for Excel/JSON/SQL/Parquet inputs, drag-and-drop upload, ML model
recommendations, a Streamlit web interface, and LLM-powered dataset
interpretation are natural next steps for this codebase.
