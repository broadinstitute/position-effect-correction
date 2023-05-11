import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def scatterplot(x, y, hue, **kwargs):
    ax = plt.gca()
    sns.scatterplot(x=x, y=y, hue=hue, ax=ax, **kwargs)

def kdeplot(x, y, hue, **kwargs):
    ax = plt.gca()
    sns.kdeplot(x=x, y=y, hue=hue ,ax=ax, **kwargs)

def remove_inner_ticklabels(fig):
    for ax in fig.axes:
        try:
            ax.label_outer()
        except:
            pass

def plot_map_per_config(
        config_df,
        config,
        x_col="mAP",
        y_col="-log(pvalue)",
        hue_col="p<0.05",
        style_col=None,
        y_log=False,
        ax_line=None,
        figsave_path="output"):
    subsets = config_df['subset'].unique()
    n_subsets = len(subsets)
    
    fig = plt.figure(figsize=(n_subsets * 5, 6))

    # Create a custom GridSpec
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
        
        # Set x-axis limits based on data range
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