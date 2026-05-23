from collections import defaultdict
import json
from pathlib import Path
import pickle
from collections import Counter
import math
import os

from lib.sanitizer import sanitizer
from constants import BM25_K1, BM25_B, CACHE_DIR

class InvertedIndex:
    def __init__(self):        
        self.index = defaultdict(set) # dictionary mapping tokens (strings) to sets of document IDs (integers)
        self.docmap: dict[int, dict] = {} # dictionary mapping document IDs to their full document objects
        
        with open('data/stopwords.txt', 'r') as file:
            stopwords = file.read().splitlines()
        self.stopwords = stopwords
        
        with open('data/movies.json', 'r') as file:
            dataset = json.load(file)
        self.movies = dataset['movies']
        
        self.term_frequencies: dict[int, Counter] = {} 
        self.doc_lengths = defaultdict()
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        
        
    def __add_document(self, doc_id: int, text: str) -> None:
        """Tokenize the input text, then add each token to the index with the document ID.
        """
        
        tokenized_words = sanitizer(text, self.stopwords)
        self.term_frequencies[doc_id] = Counter()

        # Count total number of terms per document
        self.doc_lengths[doc_id] = len(tokenized_words)

        # Log count of tokenized words
        for token in tokenized_words:
            self.index[token].add(doc_id)
            # For each token, increment its count in the Counter for that document ID
            token_counter = self.term_frequencies[doc_id] 
            token_counter.update([token])
            self.term_frequencies[doc_id] = token_counter
            
    def __get_avg_doc_length(self) -> float:
        sum_words = 0
        if len(self.docmap) == 0:
            return 0.0
        for i in self.doc_lengths:
            sum_words += self.doc_lengths[i]
        return sum_words / len(self.docmap)

    
    def get_documents(self, term) -> list[int]:
        """ Get the set of document IDs for a given token,
        and return them as a list, sorted in ascending order
            
        Keyword arguments:
        term: string, single word/token
        
        """
        # Start by getting the set of document IDs for a given token
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))
    
    def get_tf(self, doc_id, term):
        # Returns the times the token appears in the document with the given ID.
        # If the term doesn't exist in that document, return 0
        try:
            return self.term_frequencies[doc_id][term]
        except Exception as e:
            print(f"Error counting term: {e}")
            return 0
        
    def get_idf(self, term):
        # Returns the inverse document frequency.
        
        total_doc_count = len(self.docmap)
        sanitized_term = sanitizer(term, self.stopwords)
        term_match_doc_count = len(self.index[sanitized_term[0]])
        
        # Calculate the IDF
        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        return idf

    def get_bm25_idf(self, term: str) -> float:
        # Calculates a more stable IDF value via the Okapi BM25 algorithm
        tokenized_term_array = sanitizer(term, self.stopwords)
        if len(tokenized_term_array) != 1:
            raise Exception
        tokenized_term = tokenized_term_array[0]
        
        df = len(self.get_documents(tokenized_term))
        N = len(self.docmap)
        IDF = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return IDF
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        raw_term_frequency = self.get_tf(doc_id, term)
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        # saturated_tf = (raw_term_frequency * (k1 + 1)) / (raw_term_frequency + k1)
        base_tf = self.get_tf(doc_id, term)
        tf_component = (base_tf * (k1 + 1)) / (base_tf + k1 * length_norm)
        return tf_component
    
    def bm25_search(self, query, limit):
        tokens = sanitizer(query, self.stopwords)
        scores_dir = dict()
        for document in self.index:
            pass
        pass
    
    def build(self):
        # Build our cache folder
        for movie in self.movies:
            # Add to the index            
            movie_string = f"{movie['title']} {movie['description']}"
            
            # And add to the docmap
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, movie_string)
            
        
    def save(self):
        # Create cache directory if it doesn't exist
        cache_path = Path(CACHE_DIR)
        cache_path.mkdir(exist_ok=True)
        try:
            with open(os.path.join(CACHE_DIR, 'index.pkl'), 'wb') as f_index:
                pickle.dump(self.index, f_index)
            with open(os.path.join(CACHE_DIR, 'docmap.pkl'), 'wb') as f_docmap:
                pickle.dump(self.docmap, f_docmap)
            with open(os.path.join(CACHE_DIR, 'term_frequencies.pkl'), 'wb') as f_termfreq:
                pickle.dump(self.term_frequencies, f_termfreq)
            with open(os.path.join(CACHE_DIR, 'doc_lengths.pkl'), 'wb') as f_doclengths:
                pickle.dump(self.doc_lengths, f_doclengths)
            return 0
        except Exception as e:
            return e

    def load(self):
        # Load the pickle dumps from cache
        
        with open(os.path.join(CACHE_DIR,'index.pkl'), 'rb') as f_index:
            try:
                self.index = pickle.load(f_index)
            except FileNotFoundError:
                print("Error: The specified file was not found.")
            except Exception as e:
                print("Error opening index.pkl: {e}")
        
        with open(os.path.join(CACHE_DIR, 'docmap.pkl'), 'rb') as f_docmap:
            try:
                self.docmap = pickle.load(f_docmap)
            except FileNotFoundError:
                print("Error: The specified file was not found.")
            except Exception as e:
                print("Error opening docmap.pkl: {e}")
                
        with open(os.path.join(CACHE_DIR, 'term_frequencies.pkl'), 'rb') as f_termfreq:
            try:
                self.term_frequencies = pickle.load(f_termfreq)
            except FileNotFoundError:
                print("Error: The specified file was not found.")
            except Exception as e:
                print("Error opening term_frequencies.pkl: {e}")
                
        with open(os.path.join(CACHE_DIR, 'doc_lengths.pkl'), 'rb') as f_doclengths:
            try:
                self.doc_lengths = pickle.load(f_doclengths)
            except FileNotFoundError:
                print("Error: The specified file was not found.")
            except Exception as e:
                print("Error opening doc_lengths.pkl: {e}")
                

def search_for_args(query):
    db = InvertedIndex()
    db.load()
    sanitized_search_arg_tokens = sanitizer(query, db.stopwords)
    results = []
    for token in sanitized_search_arg_tokens:
        try:
            results.extend(db.get_documents(token))
            if len(results) > 5:
                return results
        except Exception as e:
            print(f"Unable to find")
    return results

def bm25_idf_command(term: str) -> float:
    # Allows us to test the `get_bm25_idf` method
    db = InvertedIndex()
    db.load()
    bm25idf = db.get_bm25_idf(term)
    return bm25idf

def bm25_tf_command(doc_id: int, term: str, k1, b):
    db = InvertedIndex()
    db.load()
    sanitized_term = sanitizer(term, db.stopwords)
    bm25tf = db.get_bm25_tf(doc_id, sanitized_term[0], k1, b)
    return bm25tf

def bm25(doc_id: int, term: str):
    db = InvertedIndex()
    db.load()
    bm25_tf = db.get_bm25_tf(doc_id, term, k1=BM25_K1, b=BM25_B)
    bm25_idf = db.get_bm25_idf(term)
    return bm25_tf * bm25_idf
