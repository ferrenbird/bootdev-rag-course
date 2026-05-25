import string
from nltk.stem import PorterStemmer
from constants import *

def sanitizer(input) -> list[str]:
    """Use this function to perform a series of sanitization steps.
    
    Limits complexity in the main search function.
    
    Keyword arguments:
    input -- string to sanitize & tokenize
    
    Outputs:
    sanatized_input -- a list of sanitized strings
    
    """
    # Start by converting everything to lowercase:
    input = input.lower()
    
    # Remove punctuation
    nopunct_lower_terms = input.translate(str.maketrans("", "", string.punctuation)).split()
    
    # Remove any stopwords & convert to stem
    stemmer = PorterStemmer()
    sanitized_input = [stemmer.stem(item) for item in nopunct_lower_terms if item not in STOPWORDS]
    
    # Return sanitized list
    return sanitized_input


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return [preprocess_text(word) for word in f.read().splitlines()]

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

STOPWORDS = load_stopwords()