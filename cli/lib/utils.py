from lib.inverted_index import InvertedIndex
from lib.sanitizer import sanitizer

def bm25_idf_command(term: str) -> float:
    # Allows us to test the `get_bm25_idf` method
    
    db = InvertedIndex()
    db.load()
    bm25idf = db.get_bm25_idf(term)
    return bm25idf

def load_stopwords() -> list:
    # Loads stopwords
    with open('data/stopwords.txt', 'r') as file:
            return file.read().splitlines()
    
def search_for_args(query):
    stopwords = load_stopwords()
    db = InvertedIndex()
    db.load()
    sanitized_search_arg_tokens = sanitizer(query, stopwords)
    results = []
    for token in sanitized_search_arg_tokens:
        try:
            results.extend(db.get_documents(token))
            if len(results) > 5:
                return results
        except Exception as e:
            print(f"Unable to find")
    return results