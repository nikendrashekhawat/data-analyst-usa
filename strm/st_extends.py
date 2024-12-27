import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def count_tokens(series) -> pd.DataFrame:
    total = len(series)
    series = series.dropna()
    arr = np.concatenate(series.values)
    unique_tokens, counts = np.unique(arr, return_counts=True)
    stacked_arr = np.column_stack((unique_tokens, counts))
    df = pd.DataFrame(stacked_arr, columns=["tokens", "count"])
    df['count'] = df['count'].astype('int64')
    df['percentage'] = (df['count'] * 100 / total).round(2)
    df['tokens'] = df['tokens'].str.title()
    return df


def plot_horizontal(y, x, xlabel=None, ylabel=None, title=None, cmap=None, xlim=None, bar_label=False, figsize=None, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if cmap is None:
        cmap = 'viridis'
    color_mapping = plt.get_cmap(cmap)
    norm = plt.Normalize(min(x), max(x))
    colors = color_mapping(norm(x))
    bars = ax.barh(y=y, width=x, color=colors, zorder=2, **kwargs)
    if bar_label:
        ax.bar_label(bars, labels=[f'{height:.1f}%' for height in x], padding=3, fontsize=4.5)
    ax.invert_yaxis()
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel, fontsize=5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_facecolor('#dbd7d2')
    ax.grid(linewidth=0.6, color='1', zorder=1)
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.tick_params(labelsize=5, bottom=False)
    return fig