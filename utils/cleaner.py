from typing import Optional
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, MWETokenizer
from reserved_keywords import (
    technical_tokens,
    soft_skills_tokens,
    education_tokens,
)

# Global Variable

# Arrays storing reserved keywords for specific categories
technical_tokens_arr = np.array(technical_tokens, dtype=np.str_)     # Technical skills keywords
softskills_tokens_arr = np.array(soft_skills_tokens, dtype=np.str_)  # Soft skills keywords
educational_tokens_arr = np.array(education_tokens, dtype=np.str_)   # Education-related keywords

# Combined keywords from all categories and filter out only unique keywords
all_keywords = np.unique(np.concatenate([technical_tokens_arr, softskills_tokens_arr, educational_tokens_arr]))

# Multi-word expressions extracted from keywords for tokenization
mwes = [tuple(phrase.split()) for phrase in all_keywords if ' ' in phrase]


class DataCleaner():
    """
    A class for cleaning and preprocessing job-related data stored in a Pandas DataFrame.

    This class provides various methods for cleaning, tokenizing, and transforming job-related 
    data. The methods include removing duplicates, filtering tokens, handling punctuation, 
    splitting tokens by categories, and extracting date and location components.

    It supports method chaining for efficient data processing.

    Attributes:
    ----------
    df : pd.DataFrame
        The DataFrame containing job-related data to be processed.

    punct : str
        Punctuation characters to be removed during text processing.

    replace_punct_with : str
        Character used to replace punctuation, defaults to a space.

    _translate_table : dict
        A translation table used for removing punctuation from text.

    _tokenizer : MWETokenizer
        A tokenizer for handling multi-word expressions using a separator.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """ 
        Initializes the DataCleaner with a DataFrame to be processed.

        Args:
        ----
        dataframe : pd.DataFrame
            The input DataFrame to be cleaned and processed.
        """
        self.df = dataframe
        self.punct = "!$%'(),-/:;?[\\]^_`{|}"
        self.replace_punct_with = ' '
        self._translate_table = str.maketrans(self.punct, self.replace_punct_with * len(self.punct))
        self._tokenizer = MWETokenizer(mwes=mwes, separator=' ')
          
              
    def _tokenize_words(self, text: str) -> np.ndarray | pd._libs.missing.NAType:
        """
        Tokenizes a string into lowercase words after removing extra spaces.

        If the text is not empty, this method removes extra spaces, converts 
        the text to lowercase, and tokenizes multi-word expressions.

        Args:
        ----
        text : str
            The text to be tokenized.

        Returns:
        -------
        np.ndarray | pd.NA
            An array of tokenized words if successful, otherwise `pd.NA`.
        """
        tokens = word_tokenize(text)
        tokens = np.array(tokens, dtype=np.str_)
        if tokens.size != 0:
            tokens = np.char.strip(tokens)
            tokens = np.char.lower(tokens)
            tokens = tokens.tolist()
            tokens = self._tokenizer.tokenize(tokens)
            tokens = np.array(tokens, dtype=np.str_)
            return tokens
        return pd.NA
    
    
    def _filter_tokens(self, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray | pd._libs.missing.NAType:
        """
        Filters tokens based on a list of allowed keywords.

        This method filters tokens from `arr1` based on whether they exist in `arr2`. 
        If `arr1` is empty or no matching tokens are found, it returns `pd.NA`.

        Args:
        ----
        arr1 : np.ndarray
            The array of tokens to filter.

        arr2 : np.ndarray
            The array of allowed tokens.

        Returns:
        -------
        np.ndarray | pd.NA
            The filtered array of tokens or `pd.NA` if no tokens are found.
        """
        if arr1 is pd.NA:
            return arr1
        filtered_arr = arr1[np.isin(arr1, arr2)]
        if filtered_arr.size != 0:
            return np.unique(filtered_arr)
        return pd.NA

    
    def remove_duplicates(self, subset: Optional[list[str]] =None, **kwargs):
        """
        Removes duplicate rows from the DataFrame based on specified columns.

        If no columns are specified, the method uses a default list of columns:
        ['title', 'company_name', 'location', 'via', 'description'].

        Args:
        ----
            subset (list[str], optional): Columns to check for duplicates. 
                Defaults to the predefined list of key columns.
            **kwargs: Additional arguments passed to `pandas.DataFrame.drop_duplicates`, 
                such as `keep`, `inplace`, or `ignore_index`.

        Returns:
        -------
            self: The current instance with duplicates removed from the DataFrame.
        """
        if subset is None:
            subset = ['title', 'company_name', 'location', 'via', 'description']
        self.df = self.df.drop_duplicates(subset=subset, ignore_index=True, **kwargs)
        return self


    def remove_columns(self, cols: list[str] | str= None):
        """
        Removes specified columns from the DataFrame.

        If no columns are specified, a default list of columns is removed:
        ['index', 'thumbnail', 'posted_at', 'job_id', 'search_term', 
        'commute_time', 'search_location', 'description_tokens'].

        Args:
        ----
            cols (list[str] | str, optional): Column name(s) to be removed. 
                If not provided, defaults to the predefined list of columns.

        Returns:
        --------
            self: The current instance with the specified columns removed from the DataFrame.
        """
        if cols is None:
            cols = ["index", "thumbnail", "posted_at", "job_id", "search_term", "commute_time", "search_location", "description_tokens"]
        self.df = self.df.drop(columns=cols)
        return self

        
    def remove_punctuations(self, col: str, table: Optional[dict]= None):
        """
        Removes punctuation from the specified column.

        Replaces punctuation characters in the given column based on a translation table.

        Args:
        ----
        col : str
            The name of the column to process.

        table : dict, optional
            A custom translation table. If not provided, the default one is used.

        Returns:
        -------
        DataCleaner
            The modified instance for method chaining.
        """
        if table is None:
            table = self._translate_table
        self.df[col] = self.df[col].str.translate(table)
        return self


    def tokenize_column(self, col: str, filter_keywords: np.ndarray= all_keywords, after_mutate: str= 'drop'):
        """
        Tokenizes and filters text from a specified column.

        Creates a new column containing tokenized and filtered words based on 
        the provided keywords.
        
        Remove the actual column from passed dataframe after tokenizing it.

        Args:
        ----
        col : str
            The name of the column to tokenize.

        filter_keywords : np.ndarray, optional
            An array of allowed keywords for filtering tokens. Default is `all_keywords`.
        
        after_mutate : str, optional {'drop', 'remain'}
            Remove actual column after tokenizing it, if `drop` is passed. To keep the 
            column in dataframe, pass `remain`.

        Returns:
        -------
        DataCleaner
            The modified instance for method chaining.
        """
        col_name = col+"_tokens"
        self.df[col_name] = self.df[col].apply(self._tokenize_words)
        self.df[col_name] = self.df[col_name].apply(self._filter_tokens, args=(filter_keywords,))
        if after_mutate == 'drop':
            self.remove_columns(col)
        return self 
    
    
    def split_tokens(self, col: str, filter_with: Optional[list[np.ndarray]]= None, expand: bool= False) -> pd.DataFrame:
        """
        Splits the tokens in a specified column into multiple columns based on given token arrays.

        Args:
        ----
        col : str
            Column to process.
                
        filter_with : Optional[list[np.ndarray]], optional 
            List of token arrays to filter the column's values.
            Defaults to predefined arrays.
                
        expand : bool, optional
            If True, adds the token columns to the original DataFrame. 
            If False, returns a new DataFrame with token columns. Defaults to False.

        Returns:
        -------
        pd.DataFrame : 
            A new DataFrame with filtered token columns or the original DataFrame 
            with added columns if `expand` is True.
        """
        if filter_with is None:
            filter_with = [technical_tokens_arr, softskills_tokens_arr, educational_tokens_arr]
        new_cols = len(filter_with)
        cols_name = ["arr_"+ str(i) for i in range(new_cols)]
        frame = {arr: self.df[col].apply(self._filter_tokens, args=(fil_arr,)) for arr, fil_arr in zip(cols_name, filter_with)}
        if expand:
            pd.concat([self.df, pd.DataFrame(frame)], axis=1)
            return self
        return pd.DataFrame(frame)


    def convert_to_datetime(self, col: str, expand= False, prefix: str= 'posted', **kwargs):
        """
        Converts a column to datetime and optionally extracts components.

        Extracts year, month, and day components if `expand` is set to True.

        Args:
        ----
        col : str
            The name of the column to convert.

        expand : bool, optional
            Whether to extract year, month, and day components.

        prefix : str, optional
            The prefix for new columns when `expand=True`.

        **kwargs : dict
            Additional keyword arguments for `pd.to_datetime`.

        Returns:
        -------
        DataCleaner
            The modified instance for method chaining.
        """
        self.df[col] = pd.to_datetime(self.df[col], **kwargs)
        if expand:
            self.df[prefix+'_year'] = self.df[col].dt.year
            self.df[prefix+'_month'] = self.df[col].dt.month
            self.df[prefix+'_day'] = self.df[col].dt.day
        return self    
    
    
    def clean_location(self, fillna_val= 'Remote USA only', repl: Optional[dict[str, str]]= None, expand = True):
        """
        Cleans and standardizes the location column.

        Fills missing values, standardizes location names, and optionally splits
        the column into city and state.

        Args:
        ----
        fillna_val : str, optional
            The default value for missing locations.

        repl : dict, optional
            A dictionary of replacements for specific locations.

        expand : bool, optional
            Whether to split the location column into city and state.

        Returns:
        -------
        DataCleaner
            The modified instance for method chaining.
        """
        if repl is None:
            repl = {'Anywhere': 'Remote Anywhere'}
        self.df['location'] = self.df['location'].str.strip()
        self.df['location'] = self.df['location'].fillna(fillna_val)
        self.df['location'] = self.df['location'].mask(self.df['location'].str.contains('United States'), other='Remote USA only')
        self.df['location'] = self.df['location'].replace(repl)
        self.df['location'] = self.df['location'].str.replace(r'\s*\(\+\d+\s+other[s]?\)', '', regex=True)
        if expand:
            loc_frame = self.df["location"].str.rsplit(',', n=1, expand=True)
            self.df['city'] = loc_frame[0]
            self.df['state'] = loc_frame[1]
            self.df['state'] = self.df['state'].fillna(self.df['city'])
        return self

    
    def clean_via(self, fillna_val: Optional[str] = None):
        """
        Cleans the 'via' column in the DataFrame.

        Fills missing values with the provided value or the column's mode 
        if no value is given. Removes the word "via" and extra spaces.

        Args:
        ----
        fillna_val : str, optional
            Value to fill missing entries. Defaults to the column's mode.

        Returns:
        -------
        DataCleaner
            The updated instance for method chaining.
        """
        if fillna_val is None:
            fillna_val = self.df["via"].mode()[0]  
        self.df["via"] = self.df["via"].fillna(fillna_val)
        self.df["via"] = self.df["via"].str.replace("via", "")
        self.df["via"] = self.df["via"].str.strip()
        return self
    
    
    def clean_schedule_type(self, repl: Optional[dict[str, str]] = None):
        """
        Cleans and standardizes the 'schedule_type' column in the DataFrame.

        If no replacement dictionary is provided, a default mapping is used 
        to standardize common schedule type values.

        Default Replacement Dictionary:
            {
                'Temp work': 'Temporary',
                'diem': 'Temporary',
                'Intern': 'Internship',
                'Volunteer': 'Volunteer',
                'Contractor': 'Contract',
                'Part-time': 'Part-time'
            }

        Missing values in 'schedule_type' are filled with 'Temporary'.

        Args:
            repl (dict[str, str], optional): A dictionary mapping substrings 
                to replacement values. If not provided, the default mapping is used.

        Returns:
            self: The current instance with the 'schedule_type' column cleaned.
        """
        if repl is None:
            repl = {
            'Temp work':'Temporary', 
            'diem':'Temporary',
            'Intern': 'Internship',
            'Volunteer':'Volunteer',
            'Contractor':'Contract',
            'Part-time':'Part-time'
            }
        series = self.df['schedule_type'].fillna('Temporary')
        for k, v in repl.items():
            series = series.mask(series.str.contains(k), v)
        self.df['schedule_type'] = series
        return self
    
    def clean_work_from_home(self):
        """
        Cleans and updates the 'work_from_home' column based on the 'description' column.

        This method checks if the 'description' column contains the phrase 
        'work from home' (case-insensitive). If the 'work_from_home' column 
        has missing values, it fills them using this check.

        Returns:
            self: The current instance with the 'work_from_home' column cleaned and updated.
        """
        mask = self.df["description"].str.lower().str.contains("work from home")
        self.df["work_from_home"] = self.df["work_from_home"].fillna(mask)
        return self
        

    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Retrieves the cleaned DataFrame.

        This method returns the processed DataFrame after all applied transformations.

        Returns:
        -------
        pd.DataFrame
            The cleaned DataFrame.
        """
        return self.df
    
if __name__ == '__main__':
    data = pd.read_csv("dataset/gsearch_jobs.csv", index_col=0)
    # ser = data["description_tokens"].head(50)
    dc = DataCleaner(data)
    ex = dc.remove_duplicates().get_cleaned_data()
    print(ex.tail())

        
    