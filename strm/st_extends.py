import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def plot_horizontal(y, x, xlabel=None, ylabel=None, title=None, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(facecolor='gainsboro', layout='constrained')
    ax.barh(y=y, width=x, **kwargs)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_facecolor('gainsboro')
    ax.grid(linestyle="--", linewidth=0.5, color='.25')
    return fig