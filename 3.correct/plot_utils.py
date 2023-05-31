from pathlib import Path
from typing import Any, Union, Optional

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_metrics(
        metrics: dict,
        metrics_columns: dict,
        groupby_columns: dict,
        results_dir: Path,
        mertrics_filename: str,
) -> pd.DataFrame:
    """
    Load metrics from parquet files and calculate mean values per groupby_columns.

    Parameters
    ----------
    metrics : dict
        Dictionary of metrics to load.
    metrics_columns : dict
        Dictionary of metrics columns to rename.
    groupby_columns : dict
        Dictionary of columns to groupby.
    results_dir : Path
        Path to results directory.
    mertrics_filename : str
        Filename of metrics parquet file.

    Returns
    -------
    metrics_df : pd.DataFrame
        Dataframe of metrics.
    """    
    metrics_dfs = []

    for subset, configs in metrics.items():
        for config, config_dir in configs.items():
            metric_cols = list(metrics_columns.keys())
            metrics_path = results_dir / config_dir / f"{mertrics_filename}.parquet"
            metrics_df = pd.read_parquet(metrics_path, columns=metric_cols + groupby_columns[config])
            metrics_df = metrics_df.groupby(groupby_columns[config])[metric_cols].mean().reset_index()
            # metrics_df.drop(groupby_columns[config], axis=1, inplace=True)
            metrics_df.rename(columns=metrics_columns, inplace=True)
            metrics_df["config"] = config
            metrics_df["subset"] = subset
            metrics_df["p<0.05"] = metrics_df["-log(pvalue)"] > 1.3
            metrics_dfs.append(metrics_df)

    metrics_df = pd.concat(metrics_dfs, axis=0)
    return metrics_df


def remove_inner_ticklabels(fig: plt.Figure) -> None:
    """
    Remove inner ticklabels from a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to remove inner ticklabels from.

    Returns
    -------
    None
    """
    for ax in fig.axes:
        try:
            ax.label_outer()
        except:
            pass


def plot_map_per_config(
        config_df: pd.DataFrame,
        config: str,
        x_col: str = "mAP",
        y_col: str = "-log(pvalue)",
        hue_col: str = "p<0.05",
        style_col: Optional[str] = None,
        y_log: bool = False,
        ax_line: Optional[Any] = None,
        figsave_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot metrics for a given config.

    Parameters
    ----------
    config_df : pd.DataFrame
        Dataframe of metrics.
    config : str
        Config to plot.
    x_col : str, optional
        Column to use as x-axis, by default "mAP".
    y_col : str, optional
        Column to use as y-axis, by default "-log(pvalue)".
    hue_col : str, optional
        Column to use as hue, by default "p<0.05".
    style_col : str, optional
        Column to use as style, by default None.
    y_log : bool, optional
        Whether to use log scale for y-axis, by default False.
    ax_line : Any, optional
        Axis line to plot, by default None.
    figsave_path : str, optional
        Path to save figure, by default None.

    Returns
    -------
    None
    """
    subsets = config_df['subset'].unique()
    n_subsets = len(subsets)
    
    fig = plt.figure(figsize=(n_subsets * 5, 6))

    # create a custom GridSpec
    gs = fig.add_gridspec(6, n_subsets)
    scatter_axes = [None] * n_subsets
    kde_axes = [None] * n_subsets

    for i in range(n_subsets):
        scatter_axes[i] = fig.add_subplot(
            gs[:5, i],
            sharex=None if i == 0 else scatter_axes[0],
            sharey=None if i == 0 else scatter_axes[0]
        )
        kde_axes[i] = fig.add_subplot(gs[5, i], sharex=scatter_axes[i], sharey=None if i == 0 else kde_axes[0])

    max_x = 0
    for i, subset in enumerate(subsets):
        subset_df = config_df[config_df['subset'] == subset]
        p_value = subset_df['p<0.05']
        print(
            config,
            subset,
            f"mmAP: {subset_df.mAP.mean():.03}",
            f"p<0.05: {p_value.mean():.03} ({p_value.sum()}/{p_value.shape[0]})",
            )

        ax_scatter = scatter_axes[i]
        ax_kde = kde_axes[i]

        sns.scatterplot(data=subset_df, x=x_col, y=y_col, hue=hue_col, style=style_col, ax=ax_scatter)
        sns.kdeplot(data=subset_df, x=x_col, hue=hue_col, ax=ax_kde, legend=False)
        
        ax_scatter.xaxis.set_major_locator(ticker.LinearLocator(7))
        ax_scatter.set_title(subset)

        if ax_line:
            ax_scatter.axhline(ax_line, ls='--')
        if y_log:
            ax_scatter.set(yscale="log")
        
        # set x-axis limits based on data range
        max_x = np.min([np.max([subset_df[x_col].max(), max_x]), 1])
        ax_scatter.set_xlim(0, max_x)
        
        if i > 0:
            ax_scatter.set_ylabel('')
            ax_kde.set_ylabel('')

    remove_inner_ticklabels(fig)
    plt.tight_layout()
    
    if figsave_path is not None:
        figsave_path = Path(figsave_path)
        figsave_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(figsave_path / f"{config}.png", bbox_inches='tight')
    
    plt.show()