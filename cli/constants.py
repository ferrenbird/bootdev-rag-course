import os

BM25_K1 = 1.5
BM25_B = 0.75
CACHE_DIR = "cache/"

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, CACHE_DIR, "movie_embeddings.npy")