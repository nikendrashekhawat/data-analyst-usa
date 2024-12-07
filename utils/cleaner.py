import numpy as np
import pandas as pd
from nltk.tokenize import MWETokenizer
from reserved_keywords import (
    technical_tokens,
    soft_skills_tokens,
    education_tokens,
)

#Global Variable
technical_tokens_arr = np.array(technical_tokens, dtype=np.str_)
softskills_tokens_arr = np.array(soft_skills_tokens, dtype=np.str_)
educational_tokens_arr = np.array(education_tokens, dtype=np.str_)

# Combine all keywords__all__ = ['tools']

all_keywords = np.unique(np.concatenate([technical_tokens_arr, softskills_tokens_arr, educational_tokens_arr]))

#Filtering multi word expressions
mwes = [tuple(phrase.split()) for phrase in all_keywords if ' ' in phrase]


class DataCleaner():

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.punct = "!$%'(),-/:;?[\\]^_`{|}"
        self.replace_punct_with = ' '
        self._translate_table = str.maketrans(self.punct, self.replace_punct_with * len(self.punct))
        self._tokenizer = MWETokenizer(mwes=mwes, separator=' ')
     
      
    def remove_punctuations(self, text: str) -> str:
        return text.translate(self._translate_table)    
    
    
    def filter_tokens(self, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray | pd._libs.missing.NAType:
        if arr1 is pd.NA:
            return arr1
        filtered_arr = arr1[np.isin(arr1, arr2)]
        if not np.any(filtered_arr):
            return pd.NA
        return np.unique(filtered_arr)    
        
              
    def tokenize_words(self, text: str) -> np.ndarray | pd._libs.missing.NAType:
        text = self.remove_punctuations(text)
        tokens = text.split()
        tokens = np.array(tokens, dtype=np.str_)
        if tokens is not None:
            tokens = np.strings.strip(tokens)
            tokens = np.strings.lower(tokens)
            tokens = tokens.tolist()
            tokens = self._tokenizer.tokenize(tokens)
            tokens = np.array(tokens, dtype=np.str_)
        else: 
            tokens = pd.NA
        return tokens
        
    
    def tokenize_column(self, col: str, filter_keywords: np.ndarray = all_keywords):
        col_name = col+"_tokens"
        self.df[col_name] = self.df[col].apply(self.tokenize_words)
        self.df[col_name] = self.df[col_name].apply(self.filter_tokens, args=(filter_keywords,))
        return self 
    
    
    def split_tokens(
            self, 
            col: str, 
            filter_with: list[np.ndarray] = [technical_tokens_arr, softskills_tokens_arr, educational_tokens_arr],
            expand: bool = False
        ) -> pd.DataFrame:

        new_cols = len(filter_with)
        print("NEW COLS", new_cols)
        cols_name = ["arr_"+ str(i) for i in range(new_cols)]
        print("COLS NAME", cols_name)
        frame = {arr: self.df[col].apply(self.filter_tokens, args=(fil_arr,)) for arr, fil_arr in zip(cols_name, filter_with)}
    
        if expand:
            return pd.concat([self.df, pd.DataFrame(frame)], axis=1)
            
        return pd.DataFrame(frame)
        
        
    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self
    
    def remove_columns(self, cols: list[str] | str = None):
        if cols is not None:
            self.df = self.df.drop(columns=cols)
        return self

    def fill_missing_values(self, value=0):
        self.df = self.df.fillna(value)
        return self

    def convert_dtype(self, col: str, dtype = None):
        if dtype is not None:
            self.df[col] = self.df[col].astype(dtype)
        return self

    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
    
if __name__ == '__main__':
    data = pd.read_csv("dataset/gsearch_jobs.csv", index_col=0)
    # ser = data["description_tokens"].head(50)
    dc = DataCleaner(data)
    ex = dc.tokenize_column("description").split_tokens("description_tokens")
    print(ex.head())

        
    