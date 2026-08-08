"""
utils.py
--------
Utility/helper functions shared across the AutoEDA Pro pipeline:
dataset loading (with encoding detection), directory management,
and simple formatting helpers.
"""

import os
import chardet
import pandas as pd


def ensure_dirs(*paths):
    """Create directories if they do not already exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def detect_encoding(file_path, sample_size=100_000):
    """
    Detect the character encoding of a CSV file by sampling
    its raw bytes with chardet.
    """
    with open(file_path, "rb") as f:
        raw_data = f.read(sample_size)
    result = chardet.detect(raw_data)
    encoding = result.get("encoding") or "utf-8"
    return encoding


SUPPORTED_EXTENSIONS = [".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".parquet"]


def _read_csv_like(source, sep=","):
    """Read a CSV/TSV-like source with encoding auto-detection.
    `source` may be a file path (str) or a file-like object (e.g. an
    uploaded file from Streamlit)."""
    is_path = isinstance(source, (str, os.PathLike))

    if is_path:
        encoding = detect_encoding(source)
    else:
        raw = source.read()
        source.seek(0)
        encoding = chardet.detect(raw[:100_000]).get("encoding") or "utf-8"

    try:
        return pd.read_csv(source, sep=sep, encoding=encoding)
    except (UnicodeDecodeError, LookupError):
        if not is_path:
            source.seek(0)
        try:
            return pd.read_csv(source, sep=sep, encoding="utf-8")
        except UnicodeDecodeError:
            if not is_path:
                source.seek(0)
            return pd.read_csv(source, sep=sep, encoding="latin-1")


def load_dataset(file):
    """
    Load a dataset of (almost) any common tabular format automatically.

    Accepts either:
      - a file path (str / os.PathLike), e.g. "input/customers.csv", or
      - a file-like / uploaded-file object with a `.name` attribute
        (e.g. a Streamlit UploadedFile from st.file_uploader).

    Supported formats: .csv, .tsv, .txt (delimited), .xlsx, .xls,
    .json, .parquet. Encoding is auto-detected for delimited text files.
    """
    is_path = isinstance(file, (str, os.PathLike))
    name = str(file) if is_path else getattr(file, "name", "uploaded_file")
    ext = os.path.splitext(name)[1].lower()

    if is_path and not os.path.exists(file):
        raise FileNotFoundError(f"Dataset not found at: {file}")

    if ext == ".csv":
        df = _read_csv_like(file, sep=",")
    elif ext in (".tsv",):
        df = _read_csv_like(file, sep="\t")
    elif ext == ".txt":
        df = _read_csv_like(file, sep=None) if False else _read_csv_like(file, sep=",")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file)
    elif ext == ".json":
        df = pd.read_json(file)
    elif ext == ".parquet":
        df = pd.read_parquet(file)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported formats: "
            f"{', '.join(SUPPORTED_EXTENSIONS)}"
        )

    return df


def get_memory_usage(df):
    """Return a human-readable memory usage string for a DataFrame."""
    bytes_used = df.memory_usage(deep=True).sum()
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_used < 1024:
            return f"{bytes_used:.2f} {unit}"
        bytes_used /= 1024
    return f"{bytes_used:.2f} TB"


def get_dataset_size(file_path):
    """Return human-readable file size on disk."""
    size = os.path.getsize(file_path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def save_text(content, path):
    """Write text content to a file, creating parent dirs if needed."""
    ensure_dirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def is_numeric_column(series):
    return pd.api.types.is_numeric_dtype(series)


def is_categorical_column(series):
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series)


def is_datetime_like(series, name_hint=""):
    """Heuristic check for whether a column looks like a date/time column."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    hint = name_hint.lower()
    keywords = ["date", "time", "timestamp", "year", "month", "day"]
    return any(k in hint for k in keywords)
