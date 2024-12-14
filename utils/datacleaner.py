import re
from typing import Optional
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, MWETokenizer
from utils.reserved_keywords import (
    technical_tokens,
    soft_skills_tokens,
    education_tokens,
)
nltk.download('punkt_tab')


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
        Initializes the DataCleaner with a DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The DataFrame to be cleaned and processed.
        """
        self.df = dataframe
        self.cleaned_file_path = None
        self.punct = "!$%'(),-/:;?[\\]^_`{|}"
        self.replace_punct_with = ' '
        self._translate_table = str.maketrans(self.punct, self.replace_punct_with * len(self.punct))
        self._tokenizer = MWETokenizer(mwes=mwes, separator=' ')
              


    def _tokenize_words(self, text: str) -> np.ndarray | pd._libs.missing.NAType:
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
        if isinstance(arr1, np.ndarray):
            if arr1.size != 0:
                filtered_arr = arr1[np.isin(arr1, arr2)]
                if filtered_arr.size != 0:
                    return np.unique(filtered_arr)
        return pd.NA
    
    


    def _extract_salary(self, extract_from: str, salary_pattern = None) -> pd.Series:
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
        temp_desc = self.df[extract_from].str.replace(r'\n', ' ', regex=True).str.replace(r'\s+', ' ', regex=True)
        if salary_pattern is None:
            number_pattern = r'\d{1,3}(?:,\d{3})?(?:[Kk])?'  # Matches numbers with optional commas or 'K|k'
            salary_pattern = fr'({number_pattern})\s?[–-]\s?({number_pattern})'  # Matches salary ranges
        salary = temp_desc.str.extract(salary_pattern, flags=re.IGNORECASE)
        formatted_salary = salary.apply(
            lambda x: f"{x[0]}-{x[1]}" if pd.notna(x[0]) and pd.notna(x[1]) else pd.NA, 
            axis=1
        )
        return formatted_salary
        


        
    def remove_duplicates(self, subset: Optional[list[str]] =None, **kwargs):
        """
        Removes duplicate rows based on specified columns.

        Parameters
        ----------
        subset : list[str], optional
            Columns to check for duplicates. Defaults to key columns {`title`, `company_name`, `description`}

        **kwargs : dict, optional
            Additional arguments for `pd.DataFrame.drop_duplicates()`.

        Returns
        -------
        DataCleaner
            The updated instance.
        """
        if subset is None:
            subset = ['title', 'company_name', 'location','description']
        self.df = self.df.drop_duplicates(subset=subset, ignore_index=True, **kwargs)
        return self


    
    def filter_data_analyst_jobs(self):
        """
        Filters job postings with 'Data Analyst' in the title (case-insensitive).

        Returns
        -------
        DataCleaner
            The updated instance with filtered job postings.
        """
        title = self.df["title"].str.lower()
        self.df = self.df[title.str.contains("data analyst")]
        self.df = self.df.reset_index(drop=True)
        return self


    def remove_columns(self, cols: list[str] | str= None):
        """
        Removes specified columns from the DataFrame.

        Parameters
        ----------
        cols : list[str] | str, optional
            Columns to remove. Defaults to a predefined list.

        Returns
        -------
        DataCleaner
            The updated instance.
        """
        if cols is None:
            cols = [
                "index", "thumbnail", "posted_at", "job_id", "search_term", "commute_time", 
                "search_location", "description_tokens", 'salary', 'salary_rate', 'salary_standardized',
                'salary_avg', 'salary_hourly', 'salary_yearly', 'salary_min', 'salary_max'
                ]
        self.df = self.df.drop(columns=cols)
        return self



        
    def remove_punctuations(self, from_col: str, table: Optional[dict]= None):
        """
        Removes punctuation from a specified column.

        Parameters
        ----------
        from_col : str
            Column name to clean.

        table : dict, optional
            Custom translation table. Uses the default table if not provided.

        Returns
        -------
        DataCleaner
            The updated instance.
        """
        if table is None:
            table = self._translate_table
        self.df[from_col] = self.df[from_col].str.translate(table)
        return self




    def tokenize_column(self, col: str, filter_keywords: np.ndarray= all_keywords, after_mutate: str= 'remain'):
        """
        Tokenizes and filters text in a specified column.

        Parameters
        ----------
        col : str
            Column to tokenize.

        filter_keywords : np.ndarray, optional
            Keywords for token filtering. Defaults to `all_keywords`.

        after_mutate : str, optional, {'drop', 'remain'}
            Whether to drop the original column. Defaults to 'remain'.

        Returns
        -------
        DataCleaner
            The updated instance.
        """
        col_name = col+"_tokens"
        self.df[col_name] = self.df[col].apply(self._tokenize_words)
        self.df[col_name] = self.df[col_name].apply(self._filter_tokens, args=(filter_keywords,))

        if after_mutate == 'drop':
            self.remove_columns(col)
        return self 
    


    
    def split_tokens(self, col_to_split: str, filter_with: Optional[list[np.ndarray]]= None, expand: bool= True, new_cols_name: Optional[list[str]]= None) -> pd.DataFrame:
        """
        Splits tokens into multiple columns based on token arrays.

        Parameters
        ----------
        col_to_split : str
            Column to split tokens from.

        filter_with : list[np.ndarray], optional
            Token arrays to filter tokens. Defaults to predefined arrays.

        expand : bool, optional
            Defaults to True. If True, adds new columns to the DataFrame. 
            Otherwise returns a `pd.DataFrame`.
        
        new_cols_name : list[str], optional
            Name given to new columns after splitting the tokens. If None,
            it will make new columns with prefix as `arr_` and numbers as suffix.

        Returns
        -------
        pd.DataFrame
            Filtered token columns or the updated DataFrame if `expand=True`.
        """
        if filter_with is None:
            filter_with = [technical_tokens_arr, softskills_tokens_arr, educational_tokens_arr]
        new_cols = len(filter_with)
        if new_cols_name is None:
            new_cols_name = ["arr_"+ str(i) for i in range(new_cols)]
        frame = {col_name: self.df[col_to_split].apply(self._filter_tokens, args=(fil_arr,)) for col_name, fil_arr in zip(new_cols_name, filter_with)}
        if expand:
            self.df = pd.concat([self.df, pd.DataFrame(frame)], axis=1)
            return self
        return pd.DataFrame(frame)




    def clean_datetime(self, col: str, expand= True, prefix: str= 'posted', **kwargs):
        """
        Converts a column to datetime and extracts components.

        Parameters
        ----------
        col : str
            Column to convert.

        expand : bool, optional
            Extracts date, year, month, and day if True. Otherwise change dtype
            of specified column to `datetime64[ns]`.

        prefix : str, optional
            Prefix for extracted components. Defaults to 'posted'.

        **kwargs : dict, optional
            Additional arguments for `pd.to_datetime()`.

        Returns
        -------
        DataCleaner
            The updated instance.
        """
        self.df[col] = pd.to_datetime(self.df[col], **kwargs)
        if expand:
            self.df[prefix+'_date'] = self.df[col].dt.date
            self.df[prefix+'_year'] = self.df[col].dt.year
            self.df[prefix+'_month'] = self.df[col].dt.month
            self.df[prefix+'_day'] = self.df[col].dt.day
        return self    
    
    


    def clean_location(self, fillna_val= 'Remote USA only', repl: Optional[dict[str, str]]= None, expand= True, after_mutate= 'drop'):
        """
        Cleans and standardizes the 'location' column.

        Parameters
        ----------
        fillna_val : str, optional
            Default value for missing locations. Defaults to 'Remote USA only'.

        repl : dict[str, str], optional
            Replacement mapping for specific locations. Uses default mapping if not provided.

        expand : bool, optional
            Splits the column into 'city' and 'state' if True. Defaults to True.
        
        after_mutate : `drop`, optional, {`drop`, `remain`}
            Remove the column if expand is True. Defaults to `drop`.

        Returns
        -------
        DataCleaner
            The updated instance.
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
            if after_mutate == 'drop':
                self.remove_columns('location')
        return self



    
    def clean_via(self, fillna_val: Optional[str] = None):
        """
        Cleans the 'via' column by filling missing values.

        Parameters
        ----------
        fillna_val : str, optional
            Value for filling missing entries. Defaults to the column's mode.

        Returns
        -------
        DataCleaner
            The updated instance.
        """
        if fillna_val is None:
            fillna_val = self.df["via"].mode()[0]  
        self.df["via"] = self.df["via"].fillna(fillna_val)
        self.df["via"] = self.df["via"].str.replace("via", "")
        self.df["via"] = self.df["via"].str.strip()
        return self
    


    
    def clean_schedule_type(self, repl: Optional[dict[str, str]] = None):
        """
        Standardizes the 'schedule_type' column.

        Parameters
        ----------
        repl : dict[str, str], optional
            Custom replacement mapping. Uses default mapping if not provided.

        Returns
        -------
        DataCleaner
            The updated instance.
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
        Updates the 'work_from_home' column based on the 'description' content.

        Returns
        -------
        DataCleaner
            The updated instance.
        """
        mask = self.df["description"].str.lower().str.contains("work from home")
        self.df["work_from_home"] = self.df["work_from_home"].fillna(mask)
        return self
     


   
    def clean_salary(self, extract_salary_from: str, expand_range=True):
        """
        Extracts and cleans salary ranges from a specified column. And fill the
        column `salary_pay` with extracted values and <NA> for missing values.

        Parameters
        ----------
        extract_salary_from : str
            Column name to extract salary from.

        expand_range : bool, optional
            Defaults to True. If True, extract `min`, `max`, `average` and 
            adds numeric salary columns to dataframe.
            

        Returns
        -------
        DataCleaner
            The updated instance.

        Notes
        -----
        - Keeps only yearly based salaries. It will remove any hourly, weekly or monthly salaries
            and fill them with <NA> after extracting salary.
        - Invalid salary values are set to `pd.NA`.
        - `salary_pay` is updated as a combined "salary_min-salary_max" string.
        """
        
        def truncate_max(x):
            if pd.isna(x):
                return pd.NA
            if len(x)<5:
                return pd.NA
            if len(x)>6:
                return x[:6]
            else:
                return x
                
        def truncate_min(x):
            if pd.isna(x):
                return pd.NA
            if len(x)>6:
                return x[:6]
            else:
                return x
                
        def clean_min(x, y):
            if pd.isna(x) or pd.isna(y):
                return pd.NA
            if len(x) < 4 and len(y) > 4:
                return x+'000'
            else:
                return x

        salary_range = self._extract_salary(extract_salary_from)
        salary_range = self.df["salary_pay"].fillna(salary_range)
        salary_range = salary_range.fillna(pd.NA)
        salary_range = salary_range.replace({'–':'-', '[,.]':'', '[Kk]':'000'}, regex=True)
        salary_range = salary_range.str.split('-', expand=True)
        salary_range = salary_range.rename({0:'salary_min', 1:'salary_max'}, axis=1)
        salary_range['salary_max'] = salary_range['salary_max'].apply(truncate_max)
        salary_range['salary_min'] = salary_range['salary_min'].apply(truncate_min)
        salary_range['salary_min'] = salary_range['salary_min'].combine(salary_range['salary_max'], clean_min)
        self.df['salary_pay'] = salary_range['salary_min'] + '-' + salary_range['salary_max']
        if expand_range:
            salary_range['salary_max'] = pd.to_numeric(salary_range['salary_max'])
            salary_range['salary_min'] = pd.to_numeric(salary_range['salary_min'])
            salary_range['salary_average'] = (salary_range['salary_max'] + salary_range['salary_min']) / 2 
            self.df = pd.concat([self.df, salary_range], axis=1)
            return self
        return self




    def save_dataset(self, path_to_save: Optional[str]=None, **kwargs):
        """
        Saves the DataFrame to a CSV file.

        Parameters
        ----------
        path : str, optional
            The file path to save the DataFrame. Default is 'dataset/cleaned_gsearch_jobs.csv'.

        **kwargs : dict, optional
            Additional arguments for `pd.DataFrame.to_csv()`.

        Returns
        -------
        None
        """
        if path_to_save is None:
            path_to_save = 'dataset/cleaned_data.csv'
        self.cleaned_file_path = path_to_save
        self.df.to_csv(path_to_save, **kwargs)
        return None




    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Retrieves the cleaned DataFrame.

        Returns
        -------
        pd.DataFrame
            The processed DataFrame.
        """
        return self.df


if __name__ == '__main__':
    pass
        
    