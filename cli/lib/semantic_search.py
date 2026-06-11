from sentence_transformers import SentenceTransformer
import numpy as np
from os import path
import json
from constants import EMBEDDINGS_PATH


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        
    def generate_embedding(self, text: str):
        if text.strip() == '':
            raise ValueError("Text input cannot be empty")
        # Remember, `text` is a string, but encode() expects a list
        embedding = self.model.encode([text], show_progress_bar=True)
        return embedding[0]
    
    def build_embeddings(self, documents: list) -> list:
        self.documents = documents
        document_list = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            document_string = f"{doc['title']}: {doc['description']}"
            document_list.append(document_string)
        self.embeddings = self.generate_embedding(document_list)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings
            
    def load_or_create_embeddings(self, documents: list):
        self.documents = documents
        document_list = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            document_string = f"{doc['title']}: {doc['description']}"
            document_list.append(document_string)
        
        if path.isfile(EMBEDDINGS_PATH):
            loaded_embeddings = np.load(EMBEDDINGS_PATH)
            if len(loaded_embeddings) == documents:
                return self.embeddings
            else:
                # Rebuild, something went wrong
                return self.build_embeddings(documents)
        else:
            # Embeddings file not found, rebuild
            return self.build_embeddings(documents)


def verify_model():
    # Create an instance of the SemanticSearch class and prints the model information
    search_model = SemanticSearch()
    print(f"Model loaded: {search_model.model}")
    print(f"Max sequence length: {search_model.model.max_seq_length}")

def embed_text(text):
    # Generate an embedding for the input text
    search_model = SemanticSearch()
    embedding = search_model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    
def verify_embeddings():
    search_model = SemanticSearch()
    with open('data/movies.json', 'r') as file:
        dataset = json.load(file)
    embeddings = search_model.load_or_create_embeddings(dataset["movies"])
    print(f"Number of docs:   {len(dataset)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")