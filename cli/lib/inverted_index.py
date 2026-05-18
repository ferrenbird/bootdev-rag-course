from collections import defaultdict
import json
from pathlib import Path
import pickle
from collections import Counter
import math

from lib.sanitizer import sanitizer

class InvertedIndex:
    def __init__(self):
        
        """
        Example of index:
        {
            'matrix': {1, 5, 10},
            'hacker': {1, 8},
            'reality': {1, 3, 7}
        }
        
        Example of docmap
        {
            4150: {
                "id": 4150,
                "title": "Memory Lane",
                "description": "NICK BOXER has just returned from war. Hes back with his old friends. His old job. And a new girl whom he met about to leap to her death from an abandoned bridge. Shes KAYLA M, a girl wrapped in mystery and hell never know her last name because on the evening that Nick was going to ask her to marry him, he finds her lying dead in the bathtub. Her wrists have been slit.\\u00a0\nIn utter despair, he attempts suicide, and finds that once his heart is stopped that he can see her, hear her, and make love to her. He can relive memories of her. And he can pay closer attention.\\u00a0 Luckily, BEN HAVEN knows CPR and rushes into the room to resuscitate Nick in time because while he is unconscious, Nick relives a vivid memory with Kayla and discovers that she didnt kill herself. Nick enlists his friends to help him build a device that is both electric chair and defibrillator. A machine with the ability to stop his heart and start it. A machine that will take him to memory lane and help him uncover all of the mystery that is Kayla M and her killer..."
            },
            4151: {
                "id": 4151,
                "title": "Freaks of Nature",
                "description": "In an all out war who will win? What will be left? In the nearly ancient right of Mortal\nCombat , when you had thought that you had seen it all - you will be overtaken, overcome . Who are the good guys? Who are the bad guys? Zombies? Vampires? Aliens? Creatures of Various report -\nFrom all walks of ghoulish life...and DEATH! Do you know what the bloodiest most horrifying thing for mom and dad was back in the day? It was the game mortal combat? Now imagine that virtually surrounding you completely.Just you and a few friends - that's all that stands between you and the rest of the world?\nOverwhelmed ? You won't be ...once you discover for yourselves who shall inherit the earth from among the horrible combative creatures in Kitchen Sink."
            },
            4152: {
                "id": 4152,
                "title": "Mies vailla menneisyytt\\u00e4",
                "description": "*************************************SPOILERS***********************As a non Finnish speaker I watched this with Finnish Subtitles to Finnish dialogue. This therefore is my limited understanding of the movie.A Man is attacked by three younger men. The attack is vicious and the first man is robbed and left for dead. He somehow staggers to the hospital where he is thought lost and left by the medical staff only to jump up and leave the place. He is robbed of his boots whilst prone by water but luck has him taken in by people that care for him rather than misuse him. He as he slowly recovers has lost his memory and therefore spends the rest of the film unable to do more than survive via the relative kindness of others.Two youngsters and their mother. A Male friend of hers too. Blind Lemon Jefferson on the sound track and other songs provide clues to the Mystery Man's love of music.The attacked man finds accommodation (at a price) essentially a container. Salvation Army personnel help him and others. And one female Salvation Army officer particular become part of his life. He starts to help the Army in their work.He adds modern or fairly modern songs to the Salvation Army's musical quartet's reportoir and a female Salvation Army leader becomes their singer. He stumbles on the fact that he may have been a welder of some skill. He has already proved he can organise and is gifted a dog. Hannibal.On a visit to the bank - it is robbed and he and a female teller are locked in a vault. The police now take an interest in this mystery man and his identity is sought via the Press.The lawyer that springs him is quite a funny turn. I have never heard a Finnish accent like that gentleman's. Almost undecipherable to me at least. The bank robber later finds him and yields up his ill gotten gains before I assume killing himself Our mystery man spread tht money around his acquaintances and later in the film we find out what part of it was for. His identity is found and he travels to meet the wife he had forgotten who was divorcing him at the time he went missing and that was now complete.It allowed both of them to continue leading their new lives. Another man for the ex-wife and Salvation Army lady for our former Mystery Man.On the way back to Helsinki the three attackers from the start of the film are encountered again but this time he is not alone. The bad men's fates are left to our imagination but I suspect the police were not troubled till later. The man and his new love exit across a railway track."
            }
        }
        """
        
        self.index = defaultdict(set) # dictionary mapping tokens (strings) to sets of document IDs (integers)
        self.docmap: dict[int, dict] = {} # dictionary mapping document IDs to their full document objects
        
        with open('data/stopwords.txt', 'r') as file:
            stopwords = file.read().splitlines()
        self.stopwords = stopwords
        
        with open('data/movies.json', 'r') as file:
            dataset = json.load(file)
        self.movies = dataset['movies']
        
        self.term_frequencies: dict[int, Counter] = {} 
        
        
    def __add_document(self, doc_id: int, text: str) -> None:
        """Tokenize the input text, then add each token to the index with the document ID.
        """
        
        tokenized_words = sanitizer(text, self.stopwords)
        self.term_frequencies[doc_id] = Counter()
        for token in tokenized_words:
            self.index[token].add(doc_id)
            # For each token, increment its count in the Counter for that document ID
            token_counter = self.term_frequencies[doc_id] 
            token_counter.update([token])
            self.term_frequencies[doc_id] = token_counter

    
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
        cache_path = Path('cache/')
        cache_path.mkdir(exist_ok=True)
        try:
            with open('cache/index.pkl', 'wb') as f_index:
                pickle.dump(self.index, f_index)
            with open('cache/docmap.pkl', 'wb') as f_docmap:
                pickle.dump(self.docmap, f_docmap)
            with open('cache/term_frequencies.pkl', 'wb') as f_termfreq:
                pickle.dump(self.term_frequencies, f_termfreq)
            return 0
        except Exception as e:
            return e

    def load(self):
        # Load the pickle dumps from cache
        
        with open('cache/index.pkl', 'rb') as f_index:
            try:
                self.index = pickle.load(f_index)
            except FileNotFoundError:
                print("Error: The specified file was not found.")
        
        with open('cache/docmap.pkl', 'rb') as f_docmap:
            try:
                self.docmap = pickle.load(f_docmap)
            except FileNotFoundError:
                print("Error: The specified file was not found.")
                
        with open('cache/term_frequencies.pkl', 'rb') as f_termfreq:
            try:
                self.term_frequencies = pickle.load(f_termfreq)
            except FileNotFoundError:
                print("Error: The specified file was not found.")
            except Exception as e:
                print("Error: {e}")