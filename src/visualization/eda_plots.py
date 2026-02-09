"""
Exploratory Data Analysis (EDA) visualization functions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


def set_plot_style(style: str = 'whitegrid', palette: str = 'Set2'):
    """Set consistent plot style."""
    sns.set_style(style)
    sns.set_palette(palette)


def plot_missing_data(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot missing data heatmap and bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        logger.info("No missing data to visualize")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Bar plot
    missing.plot(kind='barh', ax=ax1)
    ax1.set_xlabel('Number of Missing Values')
    ax1.set_title('Missing Values by Column')

    # Percentage plot
    missing_pct = (missing / len(df) * 100)
    missing_pct.plot(kind='barh', ax=ax2, color='coral')
    ax2.set_xlabel('Percentage Missing (%)')
    ax2.set_title('Missing Values Percentage')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")

    plt.close()


def plot_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    ncols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
):
    """
    Plot distributions for numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Columns to plot. If None, uses all numerical columns.
    ncols : int
        Number of columns in subplot grid
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    nrows = (len(columns) + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    for idx, col in enumerate(columns):
        if idx < len(axes):
            df[col].hist(bins=50, ax=axes[idx], edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')

    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")

    plt.close()


def plot_boxplots(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    ncols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
):
    """
    Plot boxplots for numerical columns to identify outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Columns to plot
    ncols : int
        Number of columns in subplot grid
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    nrows = (len(columns) + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    for idx, col in enumerate(columns):
        if idx < len(axes):
            df.boxplot(column=col, ax=axes[idx])
            axes[idx].set_title(f'Boxplot of {col}')
            axes[idx].set_ylabel(col)

    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")

    plt.close()


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot correlation heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Columns to include in correlation
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    if columns is None:
        df_corr = df.select_dtypes(include=[np.number])
    else:
        df_corr = df[columns]

    corr_matrix = df_corr.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={'shrink': 0.8}
    )
    plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")

    plt.close()


def plot_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: Optional[str] = None,
    freq: str = 'D',
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
):
    """
    Plot time series of transactions or values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    date_col : str
        Date column name
    value_col : str, optional
        Value column to aggregate. If None, counts rows.
    freq : str
        Frequency for aggregation ('D', 'W', 'M')
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    df_ts = df.copy()

    # Ensure date column is datetime
    df_ts[date_col] = pd.to_datetime(df_ts[date_col])

    # Set date as index
    df_ts = df_ts.set_index(date_col)

    # Aggregate
    if value_col:
        ts_data = df_ts[value_col].resample(freq).sum()
        ylabel = f'Total {value_col}'
    else:
        ts_data = df_ts.resample(freq).size()
        ylabel = 'Number of Transactions'

    plt.figure(figsize=figsize)
    ts_data.plot(linewidth=2)
    plt.title(f'Time Series ({freq} frequency)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")

    plt.close()


def plot_top_n(
    df: pd.DataFrame,
    column: str,
    top_n: int = 10,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot top N values for a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name
    top_n : int
        Number of top values to show
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    top_values = df[column].value_counts().head(top_n)

    plt.figure(figsize=figsize)
    top_values.plot(kind='barh')
    plt.title(f'Top {top_n} {column}', fontsize=14, fontweight='bold')
    plt.xlabel('Count')
    plt.ylabel(column)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")

    plt.close()


