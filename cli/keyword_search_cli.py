import json
import argparse

from lib.inverted_index import InvertedIndex
from lib.sanitizer import sanitizer

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    build_parser = subparsers.add_parser("build", help="Build database")

    args = parser.parse_args()
    
    # Load our dataset
    results = []
    with open('data/movies.json', 'r') as file:
        dataset = json.load(file)
    
    with open('data/stopwords.txt', 'r') as file:
        stopwords = file.read().splitlines()

    match args.command:
        case "build":
            # Build our Inverted Index
            db = InvertedIndex()
            db.build()
            
            # Save to list
            db.save()
            
            # TODO: Add logger statements on success?

        
        case "search":
            # Perform the search
            print(f"Searching for: {args.query}")
            
            # Scan through movies
            movie_counter = 0
            
            # We're now going to use our InvertedIndex data set via load()
            db = InvertedIndex()
            try:
                db.load()
            except Exception as e:
                print(f"Exception encountered: {e}")
                return
            
            sanitized_search_arg_tokens = sanitizer(args.query, stopwords)
            results = []
            for token in sanitized_search_arg_tokens:
                try:
                    results.extend(db.get_documents(token))
                    if len(results) > 5:
                        break
                except Exception as e:
                    print(f"Unable to find")

            for result in results:
                movie_counter += 1
                print(f"{db.docmap[result]['id']}. {db.docmap[result]['title']}")
                if movie_counter == 5:
                    break 
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()