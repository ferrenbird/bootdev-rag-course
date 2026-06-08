from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')


def verify_model():
    # Create an instance of the SemanticSearch class and prints the model information
    search_model = SemanticSearch()
    print(f"Model loaded: {search_model.model}")
    print(f"Max sequence length: {search_model.model.max_seq_length}")
    pass

'''
from sentence_transformers import SentenceTransformer

# Load the model (downloads automatically the first time)
model = SentenceTransformer('all-MiniLM-L6-v2')

print(f"Model loaded: {model}")
print(f"Max sequence length: {model.max_seq_length}")

model.encode(text)
'''