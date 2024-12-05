from typing import Union
import numpy as np
import pandas as pd
from typing import Callable
from nltk.tokenize import MWETokenizer
from utils.reserved_keywords import (
    technical_tokens,
    soft_skills_tokens,
    education_tokens,
)


technical_tokens_arr = np.array(technical_tokens, dtype=np.str_)
softskills_tokens_arr = np.array(soft_skills_tokens, dtype=np.str_)
educational_tokens_arr = np.array(education_tokens, dtype=np.str_)

# Combine all multi-word expressions (MWEs) and keywords
all_keywords = np.unique(np.concatenate([technical_tokens_arr, softskills_tokens_arr, educational_tokens_arr]))

# Translation table for replacing punctuation with whitespace
punctuations = "!$%'(),-./:;?[\\]^_`{|}"
translate_table = str.maketrans(punctuations, ' ' * len(punctuations))

# Tokenize MWEs
mwes = [tuple(phrase.split()) for phrase in all_keywords if ' ' in phrase]
mwe_tokenizer = MWETokenizer(mwes=mwes, separator=' ')

#Function to remove punctuation
remove_punctuation: Callable[[str], str] = lambda text: text.translate(translate_table)



def get_word_tokens(text: str, tokenizer: MWETokenizer = mwe_tokenizer, keywords: np.ndarray[np.str_] = all_keywords):
    text = remove_punctuation(text)
    tokens = text.split()
    tokens = np.array(tokens, dtype=np.str_)
    if tokens is None:
        return pd.NA
    else:
        tokens = np.strings.strip(tokens)
        tokens = np.strings.lower(tokens)
        tokens = tokens.tolist()
        new_tokens = tokenizer.tokenize(tokens)
        new_tokens = np.array(new_tokens, dtype=np.str_)
        filtered_tokens = new_tokens[np.isin(new_tokens, all_keywords)]
        if not np.any(filtered_tokens):
            return pd.NA
        else: 
            return np.unique(filtered_tokens)
        
        
                    
def get_technical_tokens(tokens_array):
    
    if tokens_array is pd.NA:
        return pd.NA
    filtered_tokens = tokens_array[np.isin(tokens_array, technical_tokens_arr)]
    
    if not np.any(filtered_tokens):
        return pd.NA
    else:
        return filtered_tokens
    


def get_softskills_tokens(tokens_array):
    if tokens_array is pd.NA :
        return pd.NA
    
    filtered_tokens = tokens_array[np.isin(tokens_array, softskills_tokens_arr)]
    
    if not np.any(filtered_tokens):
        return pd.NA
    else:
        return filtered_tokens



def get_educational_tokens(tokens_array):
    if tokens_array is pd.NA:
        return pd.NA
    
    filtered_tokens = tokens_array[np.isin(tokens_array, education_tokens)]
    
    if not np.any(filtered_tokens):
        return pd.NA
    else:
        return filtered_tokens
    



