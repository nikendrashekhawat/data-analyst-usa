import numpy as np
from src._util.reserved_keywords import (
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


tokens_to_normalize = {
    'bachelors':'bachelor',
    "bachelor's":'bachelor', 
    'masters':'master',
    "master's":'master',
    'doctorate':'phd',
    'postgraduate':'diploma',
    'bs degree':'bachelor',
    'ba degree':'bachelor',
    'b s degree':'bachelor',
    'b a degree':'bachelor',
    'information technology':'bachelor',
    'computer science':'bachelor',
    'accredited university':'bachelor',
    'professional certificate':'certification',
    'technical degree':'bachelor', 
    'advanced degree':'bachelor',
    'mathematics':'bachelor',
    'economics':'bachelor',
    'dashboard':'dashboards',
    }
