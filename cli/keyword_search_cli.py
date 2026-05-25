import argparse
from lib.keyword_search import *
from constants import BM25_K1, BM25_B

def main() -> None:
    
    # Generic CLI Parser & subparser
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Search Parser
    search_parser = subparsers.add_parser(
        "search", help="Search movies using BM25"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    
    # Build Parser
    build_parser = subparsers.add_parser(
        "build", help="Build database"
    )
    
    # TF Parser
    tf_parser = subparsers.add_parser(
        "tf", help="Gets term frequency"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency of")
    
    # IDF Parser
    idf_parser = subparsers.add_parser(
        "idf", help="Calculate Inverse Document Frequency"
    )
    idf_parser.add_argument("term", type=str, help="Term to get inverse document frequency of")
    
    # TFIDF Parser
    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Calculate `Term & Inverse Document` Frequency"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get inverse document frequency of")
    
    # BM25 Parser
    bm25_parser = subparsers.add_parser(
        "bm25idf", help="Calculate `Okapi BM25` IDF"
    )
    bm25_parser.add_argument("term", type=str, help="Term to get inverse document frequency of")
    
    # BM25_TF Parser
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs='?', default=5, help="Max number of movies to return")

    args = parser.parse_args()
    
    with open('data/stopwords.txt', 'r') as file:
        stopwords = file.read().splitlines()

    match args.command:
        case "build":
            # Build our Inverted Index
            db = InvertedIndex()
            db.build()
            
            # Save to list
            save_status = db.save()
            
            if (save_status):
                print(f"There was an error with saving - {save_status}")
            else:
                print(f"Inverted index built successfully")
                
            
        case "tf":
            # Finds term frequency for a given term in the specified document.
            # If the term doesn't exist in that document, it should print "0".
            
            print(f"Counting {args.term} across document #{args.doc_id}")
            db = load_dataset()
            try:
                term_count = db.get_tf(args.doc_id, args.term)
            except Exception as e:
                print(f"Exception encountered: {e}")
                return
            print(f"The term `{args.term}` appeared {term_count} time(s) in document #{args.doc_id}")
        
        case "idf":
            # Calculate the IDF for a given term
            db = load_dataset()
            
            try:
                idf = db.get_idf(args.term)
            except Exception as e:
                print(f"Exception encountered: {e}")
                return
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}") 
            return
                
        case "bm25idf":
            # Calculate BM25 Inverse Document frequency
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
                
        case "tfidf":
            # Calculates the TFxIDF value for a given term in a given document
            # It should take a document ID and a term as arguments.
            # If the term doesn't exist in that document, it should print "0".
            print(f"Counting {args.term} across document #{args.doc_id}")
            db = load_dataset()
            try:
                # Calculate TF
                tf = db.get_tf(args.doc_id, args.term)                
                # Calculate IDF
                idf = db.get_idf(args.term)
                # Calculate TFIDF
                tf_idf = tf * idf
                print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

            except Exception as e:
                print(f"Exception encountered: {e}")
                
        case "bm25tf":
            # Calculate BM25 Inverse Document frequency
            # Pass in the terms raw - we'll process later
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
                
        case "search":
            # Perform the search
            print(f"Searching for: {args.query}")
            
            # We're now going to use our InvertedIndex data set via load()
            db = load_dataset()
            
            # Pass in the terms raw - we'll process later
            results = search_for_args(args.query)
            
            # Scan through movies
            movie_counter = 0
            for result in results:
                movie_counter += 1
                print(f"{db.docmap[result]['id']}. {db.docmap[result]['title']}")
                if movie_counter == 5:
                    break
                
        case "bm25search":
            db = load_dataset()
            
            # Pass in the terms raw - we'll process later
            results = db.bm25_search(args.query, args.limit)
            i = 1
            for result in results:
                print(f"{i}. ({db.docmap[result]['id']}) {db.docmap[result]['title']} - Score: {results[result]:.2f}")
                i += 1
            return

        case _:
            parser.print_help()

def load_dataset():
    # Returns dataset
    db = InvertedIndex()
    try:
        db.load()
        return db
    except Exception as e:
        print(f"Exception encountered: {e}")
        return

if __name__ == "__main__":
    main()