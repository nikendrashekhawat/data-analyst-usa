import re
from nltk.tokenize import MWETokenizer

from utils.reserved_keywords import (
    technical_skills,
    soft_skills,
    technical_mwe,
    soft_skills_mwe,
    education_requirement,
    education_mwe
)

# Combine all multi-word expressions (MWEs) and keywords
combined_mwes = technical_mwe + soft_skills_mwe + education_mwe
keywords = set(combined_mwes + technical_skills + soft_skills + education_requirement)

# Translation table for replacing punctuation with whitespace
punctuations = "!$%'(),-./:;?[\\]^_`{|}"
translate_table = str.maketrans(punctuations, ' ' * len(punctuations))


# Tokenize MWEs
mwes = [tuple(phrase.split()) for phrase in combined_mwes]
mwe_tokenizer = MWETokenizer(mwes=mwes, separator=' ')

#Function to remove punctuation
remove_punctuation = lambda text: text.translate(translate_table)

# Function to remove extra spaces
remove_extra_spaces = lambda text: re.sub(r'\s+', ' ', text).strip()
   

def get_word_tokens(text: str) -> list[str]:
    
    text = remove_punctuation(text)
    text = remove_extra_spaces(text)
    text = text.lower()
    
    tokens = text.split()
    
    new_tokens = mwe_tokenizer.tokenize(tokens)
    
    filtered_tokens = [token for token in new_tokens if token in keywords] 

    return list(set(filtered_tokens))




