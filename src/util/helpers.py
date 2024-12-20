import re
from typing import Optional
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, MWETokenizer
from src.util.keywords import mwes
from src.util.keywords import tokens_to_normalize

nltk.download('punkt_tab')

tokenizer = MWETokenizer(mwes=mwes, separator=' ')



def tokenize_words(text: str) -> Optional[np.ndarray]:
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




def normalize_tokens(arr: np.ndarray, repl: Optional[dict[str, str]]= None) -> Optional[np.ndarray]:
    if not isinstance(arr, np.ndarray):
        return pd.NA
    if repl is None:
        repl = tokens_to_normalize
    for k, v in repl.items():
        arr = np.char.replace(arr, k, v)
    return np.unique(arr)




def filter_tokens(arr1: np.ndarray, arr2: np.ndarray) -> Optional[np.ndarray]:
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




def extract_salary(extract_from: pd.Series, salary_pattern = None):
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
        number_pattern = r'\d{1,3}(?:,\d{3})?(?:\.\d+)?[Kk]?'  # Matches numbers with optional commas or 'K|k'
        salary_pattern = fr'({number_pattern})\s?[–-]\s?({number_pattern})'  # Matches salary ranges
    salary = temp_series.str.extract(salary_pattern, flags=re.IGNORECASE)
    salary = salary.replace(',', '', regex=True)
    salary = salary.rename({0:'min', 1:'max'}, axis=1)
    salary['min'] = salary['min'].apply(convert_salary_to_number)
    salary['max'] = salary['max'].apply(convert_salary_to_number)
    return salary


def convert_salary_to_number(salary_str):
    """
    Convert a salary string to a numeric value.

    Parameters
    ----------
    salary_str : str
        The salary string, which may include 'K' for thousands.

    Returns
    -------
    float
        The numeric salary value. Returns NaN if the input is NaN.
    """
    if pd.isna(salary_str):
        return np.nan
    salary_str = salary_str.lower()
    if ('k' in salary_str) and (len(salary_str) > 4):
        salary_str = salary_str[:-1]
    if 'k' in salary_str:
         return float(salary_str[:-1]) * 1000
    return float(salary_str)



def fix_minimum_salary(minimum, maximum):
    """
    Adjust the minimum salary based on contextual rules.

    Parameters
    ----------
    minimum : float
        The minimum salary value.
    maximum : float
        The maximum salary value.

    Returns
    -------
    float
        The corrected minimum salary. Returns NaN if adjustments are invalid.
    """
    if pd.isna(minimum) and pd.isna(maximum):
        return np.nan
    if pd.isna(minimum) and pd.notna(maximum):
        minimum = maximum
    if minimum < 1000 and maximum > 10000:
        if minimum < 10:
            minimum = minimum * 10000
        else:
            minimum = minimum * 1000
    if minimum <= 10000:
        return np.nan
    return minimum
  
  

def fix_maximum_salary(minimum, maximum):
    """
    Adjust the maximum salary based on contextual rules.

    Parameters
    ----------
    minimum : float
        The minimum salary value.
    maximum : float
        The maximum salary value.

    Returns
    -------
    float
        The corrected maximum salary. Returns NaN if adjustments are invalid.
    """
    if pd.isna(minimum) and pd.isna(maximum):
        return np.nan
    if pd.isna(maximum) and pd.notna(minimum):
        maximum = minimum 
    if maximum < 1000 and minimum > 10000:
        if maximum < 10:
            maximum = maximum * 10000
        else:
            maximum = maximum * 1000
    if maximum < minimum:
            maximum = minimum
    if maximum <= 20000:
        return np.nan
    return maximum
