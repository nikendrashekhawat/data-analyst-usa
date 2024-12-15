import re
from typing import Optional
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, MWETokenizer
from src._util.keywords import mwes
from src._util.keywords import tokens_to_normalize

nltk.download('punkt_tab')

tokenizer = MWETokenizer(mwes=mwes, separator=' ')



def _tokenize_words(text: str) -> Optional[np.ndarray]:
    """
    Tokenizes a string into lowercase words after removing extra spaces.

    Parameters
    ----------
    text : str
        The text to tokenize.

    Returns
    -------
    np.ndarray | pd.NA
        An array of tokenized words or `pd.NA` if empty.
    """
    tokens = word_tokenize(text)
    if not tokens:
        return pd.NA
    tokens = np.array(tokens, dtype=np.str_)
    tokens = np.char.strip(tokens)
    tokens = np.char.lower(tokens)
    tokens = tokenizer.tokenize(tokens)
    tokens = np.array(tokens, dtype=np.str_)
    return tokens




def _normalize_tokens(arr: np.ndarray, repl: Optional[dict[str, str]]= None) -> Optional[np.ndarray]:
    if not isinstance(arr, np.ndarray):
        return pd.NA
    if repl is None:
        repl = tokens_to_normalize
    for k, v in repl.items():
        arr = np.char.replace(arr, k, v)
    return np.unique(arr)




def _filter_tokens(arr1: np.ndarray, arr2: np.ndarray) -> Optional[np.ndarray]:
    """
    Filters tokens in `arr1` based on matches in `arr2`.

    Parameters
    ----------
    arr1 : np.ndarray
        Tokens to filter.

    arr2 : np.ndarray
        Allowed tokens for filtering.

    Returns
    -------
    np.ndarray | pd.NA
        Filtered tokens or `pd.NA` if none match.
    """
    if isinstance(arr1, np.ndarray) and arr1.size:
        filtered_arr = arr1[np.isin(arr1, arr2)]
        if filtered_arr.size:
            return np.unique(filtered_arr)
    return pd.NA




def _extract_salary(extract_from: pd.Series, salary_pattern: Optional[str] = None) -> pd.Series:
    """
    Extracts salary ranges from a specified column.

    Parameters
    ----------
    extract_from : str
        Column name to extract salary from.

    salary_pattern : str, optional
        Custom regex pattern. Defaults to a predefined salary range pattern.

    Returns
    -------
    pd.Series
        Extracted and formatted salary ranges.
    """
    temp_series = extract_from.copy()
    temp_series = temp_series.str.replace(r'\n', ' ', regex=True).str.replace(r'\s+', ' ', regex=True)
    if salary_pattern is None:
        number_pattern = r'\d{1,3}(?:,\d{3})?(?:[Kk])?'  # Matches numbers with optional commas or 'K|k'
        salary_pattern = fr'({number_pattern})\s?[–-]\s?({number_pattern})'  # Matches salary ranges
    salary = temp_series.str.extract(salary_pattern, flags=re.IGNORECASE)
    formatted_salary = salary.apply(
        lambda x: f"{x[0]}-{x[1]}" if pd.notna(x[0]) and pd.notna(x[1]) else pd.NA, 
        axis=1
    )
    return formatted_salary



def truncate_max_salary(x: Optional[str]) -> Optional[str]:
    if pd.isna(x):
        return pd.NA
    if len(x)<5:
        return pd.NA
    if len(x)>6:
        return x[:6]
    else:
        return x


   
def truncate_min_salary(x: Optional[str]) -> Optional[str]:
    if pd.isna(x):
        return pd.NA
    if len(x)>6:
        return x[:6]
    else:
        return x
   
   
   
     
def clean_min_salary(x: Optional[str], y: Optional[str]) -> Optional[str]:
    if pd.isna(x) or pd.isna(y):
        return pd.NA
    if len(x) < 4 and len(y) > 4:
        return x+'000'
    else:
        return x