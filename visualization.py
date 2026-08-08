"""
visualization.py
-----------------
Automated chart generation for AutoEDA Pro. Produces professional
Matplotlib/Seaborn visualizations and saves them as PNG files under
the charts/ directory:

  histogram.png, boxplot.png, countplot.png, scatterplot.png,
  heatmap.png, pairplot.png, barchart.png, piechart.png,
  linechart.png (only if a datetime-like column is present)
"""

import logging
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from utils import is_datetime_like

sns.set_theme(style="whitegrid")

logger = logging.getLogger("autoeda.visualization")


class ChartGenerator:
    def __init__(self, df, output_dir="charts"):
        self.df = df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.generated_files = []

    def _save(self, fig, filename):
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.generated_files.append(path)
        return path

    # ------------------------------------------------------------------
    def histogram(self, filename="histogram.png", max_cols=6):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
        if not numeric_cols:
            return None

        n = len(numeric_cols)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)

        for i, col in enumerate(numeric_cols):
            sns.histplot(self.df[col].dropna(), kde=True, ax=axes[i], color="#4C72B0")
            axes[i].set_title(f"Distribution of {col}")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.tight_layout()
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    def boxplot(self, filename="boxplot.png", max_cols=8):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
        if not numeric_cols:
            return None

        fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols) * 1.2), 6))
        sns.boxplot(data=self.df[numeric_cols], ax=ax, palette="Set2")
        ax.set_title("Box Plot - Outlier Detection")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    def countplot(self, filename="countplot.png", max_cols=4, max_categories=15):
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols = [c for c in cat_cols if self.df[c].nunique() <= max_categories][:max_cols]
        if not cat_cols:
            return None

        n = len(cat_cols)
        ncols = min(2, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)

        for i, col in enumerate(cat_cols):
            order = self.df[col].value_counts().index
            sns.countplot(y=self.df[col], order=order, ax=axes[i], palette="viridis")
            axes[i].set_title(f"Frequency of {col}")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.tight_layout()
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    def scatterplot(self, filename="scatterplot.png"):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return None

        corr = self.df[numeric_cols].corr().abs()
        corr_arr = corr.to_numpy(copy=True)
        np.fill_diagonal(corr_arr, 0)
        corr = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)
        x_col, y_col = corr.stack().idxmax()

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.scatterplot(x=self.df[x_col], y=self.df[y_col], ax=ax, alpha=0.7, color="#DD8452")
        ax.set_title(f"{x_col} vs {y_col}")
        fig.tight_layout()
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    def correlation_heatmap(self, filename="heatmap.png"):
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None

        fig, ax = plt.subplots(figsize=(max(6, numeric_df.shape[1]), max(5, numeric_df.shape[1] * 0.8)))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    def pairplot(self, filename="pairplot.png", max_cols=5):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
        if len(numeric_cols) < 2:
            return None

        g = sns.pairplot(self.df[numeric_cols].dropna(), diag_kind="kde")
        g.fig.suptitle("Pair Plot", y=1.02)
        path = os.path.join(self.output_dir, filename)
        g.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(g.fig)
        self.generated_files.append(path)
        return path

    # ------------------------------------------------------------------
    def barchart(self, filename="barchart.png", max_categories=15):
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols = [c for c in cat_cols if self.df[c].nunique() <= max_categories]
        if not cat_cols:
            return None
        col = cat_cols[0]

        fig, ax = plt.subplots(figsize=(8, 5))
        self.df[col].value_counts().plot(kind="bar", ax=ax, color="#55A868")
        ax.set_title(f"Bar Chart - {col} Frequency")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    def piechart(self, filename="piechart.png", max_categories=8):
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols = [c for c in cat_cols if self.df[c].nunique() <= max_categories]
        if not cat_cols:
            return None
        col = cat_cols[0]

        fig, ax = plt.subplots(figsize=(6, 6))
        self.df[col].value_counts().plot(
            kind="pie", autopct="%1.1f%%", ax=ax, colors=sns.color_palette("pastel")
        )
        ax.set_ylabel("")
        ax.set_title(f"Proportion of {col}")
        fig.tight_layout()
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    def linechart(self, filename="linechart.png"):
        datetime_cols = [c for c in self.df.columns if is_datetime_like(self.df[c], name_hint=c)]
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not datetime_cols or not numeric_cols:
            return None

        date_col = datetime_cols[0]
        y_col = numeric_cols[0]
        temp = self.df.copy()
        try:
            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        except Exception:
            return None
        temp = temp.dropna(subset=[date_col]).sort_values(date_col)
        if temp.empty:
            return None

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(temp[date_col], temp[y_col], color="#4C72B0")
        ax.set_title(f"{y_col} over Time")
        ax.set_xlabel(date_col)
        ax.set_ylabel(y_col)
        fig.autofmt_xdate()
        fig.tight_layout()
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    def violinplot(self, filename="violinplot.png", max_cols=8):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
        if not numeric_cols:
            return None

        fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols) * 1.2), 6))
        sns.violinplot(data=self.df[numeric_cols], ax=ax, palette="Set3")
        ax.set_title("Violin Plot - Distribution & Density")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    def kdeplot(self, filename="kdeplot.png", max_cols=6):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
        if not numeric_cols:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        for col in numeric_cols:
            sns.kdeplot(self.df[col].dropna(), ax=ax, label=col, fill=False)
        ax.set_title("KDE Plot - Numeric Feature Density")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    def distribution_plot(self, filename="distribution.png", max_cols=6):
        """Distribution plot (histogram + KDE) for the numeric column with
        the highest absolute skewness, as a focused complement to histogram()."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return None
        skews = {c: abs(self.df[c].dropna().skew()) for c in numeric_cols if self.df[c].dropna().shape[0] > 1}
        if not skews:
            return None
        col = max(skews, key=skews.get)

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(self.df[col].dropna(), kde=True, ax=ax, color="#8172B2")
        ax.set_title(f"Distribution of {col} (skewness = {skews[col]:.2f})")
        fig.tight_layout()
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    def qq_plot(self, filename="qqplot.png"):
        """QQ plot (against a normal distribution) for the most-skewed numeric column."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return None
        skews = {c: abs(self.df[c].dropna().skew()) for c in numeric_cols if self.df[c].dropna().shape[0] > 1}
        if not skews:
            return None
        col = max(skews, key=skews.get)

        fig, ax = plt.subplots(figsize=(6, 6))
        stats.probplot(self.df[col].dropna(), dist="norm", plot=ax)
        ax.set_title(f"QQ Plot - {col}")
        fig.tight_layout()
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    def missing_value_heatmap(self, filename="missing_heatmap.png"):
        if self.df.isnull().sum().sum() == 0:
            return None

        fig, ax = plt.subplots(figsize=(max(6, self.df.shape[1] * 0.6), 5))
        sns.heatmap(self.df.isnull(), cbar=False, cmap="rocket_r", ax=ax, yticklabels=False)
        ax.set_title("Missing Value Heatmap")
        fig.tight_layout()
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    def generate_all(self, include_extended=True):
        """Generate every applicable chart and return list of saved paths.
        Set include_extended=False to keep the original chart set only."""
        self.histogram()
        self.boxplot()
        self.countplot()
        self.scatterplot()
        self.correlation_heatmap()
        self.pairplot()
        self.barchart()
        self.piechart()
        self.linechart()
        if include_extended:
            self.violinplot()
            self.kdeplot()
            self.distribution_plot()
            self.qq_plot()
            self.missing_value_heatmap()
        return self.generated_files
