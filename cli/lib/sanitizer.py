import string
from nltk.stem import PorterStemmer

def sanitizer(input, stopwords) -> list[str]:
    """Use this function to perform a series of sanitization steps.
    
    Limits complexity in the main search function.
    
    Keyword arguments:
    input -- string to sanitize & tokenize
    
    Outputs:
    sanatized_input -- a list of sanitized strings
    
    """
    # First, make a punctuation translate table with maketrans()
    punct_dict = {}
    for char in string.punctuation:
        punct_dict[char] = ' '
        
    # Then, use that dictionary to create a translation table (to remove punctuation)
    punct_table = str.maketrans(punct_dict)
    
    # Then use this translate table in translate()
    sanitized_input = input.translate(punct_table).split()
    
    # Convert all strings in list to lowercase
    sanitized_input = [input.lower() for input in sanitized_input]
    
    # Remove any stopwords & convert to stem
    stemmer = PorterStemmer()
    sanitized_input = [stemmer.stem(item) for item in sanitized_input if item not in stopwords]
    
    # Return sanitized list
    return sanitized_input
