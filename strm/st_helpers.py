import numpy as np
import pandas as pd


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