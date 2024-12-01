from typing import List
from functools import cache
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.mwe import MWETokenizer
from nltk.corpus import stopwords

nltk.download('punkt_tab')


@cache
def get_word_tokens(description: str) -> List[str]:
    
    tokens = word_tokenize(description.lower())
    
    stop_words = set(stopwords.words('english'))
    
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    return filtered_tokens