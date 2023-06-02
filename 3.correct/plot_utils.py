import string
from pathlib import Path
from typing import Any, Optional, Sequence, Union
from typing import SupportsFloat as Numeric

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
        figsave_path: str = "output"
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
        Path to save figure, by default "output".

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
        max_x = np.fmin(subset_df[x_col].max(), 1)
        ax_scatter.set_xlim(0, max_x)
        
        if i > 0:
            ax_scatter.set_ylabel('')
            ax_kde.set_ylabel('')

    remove_inner_ticklabels(fig)
    plt.tight_layout()
    figsave_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(figsave_path / f"{config}_well_mean_correction.png", bbox_inches='tight')
    plt.show()


def add_well_location(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add x and y location of well to dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe of features.

    Returns
    -------
    pd.DataFrame
        Dataframe of features with x and y location of well.
    """
    well_list = [(num, lit_i + 1, f"{lit}{num:02}") for lit_i, lit in enumerate(list(string.ascii_uppercase)[15::-1]) for num in range(1, 25)]
    well_list = pd.DataFrame(well_list, columns=["x_loc", "y_loc", "Metadata_Well"])
    data = data.merge(well_list, on="Metadata_Well")
    return data


def plot_mean_feature_per_well(
        data: pd.DataFrame,
        feature: str,
        prefix: str = "",
        colormap: str = "PRGn",
        colormap_range: Optional[Sequence[Numeric]]=None,
        figsave_path: Optional[Union[Path, str]] = None
) -> None:
    """
    Plot feature mean per well.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe of features.
    feature : str
        Feature to plot.
    colormap : str, optional
        Colormap to use, by default "PRGn".
    colormap_range : Sequence[Numeric, Numeric], optional
        Range of colormap, by default None.

    Returns
    -------
    None
    """
    data = data.groupby("Metadata_Well")[feature].mean().reset_index(drop=False)
    data = add_well_location(data)

    colormap_range = colormap_range or (data[feature].min(), data[feature].max())
    colormap_norm = plt.Normalize(*colormap_range)

    fig, ax = plt.subplots(figsize=(10,6))
    scatter = sns.scatterplot(data=data, x="x_loc", y="y_loc", hue=feature, hue_norm=colormap_norm, palette="PRGn", s=250, legend=False, marker="s")
    for _, row in data.iterrows():
        ax.text(row['x_loc'], row['y_loc'], row['Metadata_Well'], color='lightgrey', ha='center', va='center', weight='ultralight', size=7)

    ax.set_aspect('equal')
    ax.set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)
    ax.set_title(f"{prefix} {feature}")

    colormap_range = colormap_range or (data[feature].min(), data[feature].max())
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=colormap_norm)
    sm.set_array([])
    fig.colorbar(sm)

    if figsave_path is not None:
        figsave_path = Path(figsave_path)
        figsave_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(figsave_path / f"{prefix}_{feature}_mean_per_well.png", bbox_inches='tight')

    plt.show()