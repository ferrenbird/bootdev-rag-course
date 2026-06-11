import argparse
from lib.semantic_search import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build Parser
    verify_parser = subparsers.add_parser(
        "verify", help="Verify model initialization"
    )
    
    # Embed Text Parser
    embed_parser = subparsers.add_parser(
        "embed_text", help="Verify model initialization"
    )
    embed_parser.add_argument("text", type=str, help="Text to generate embedding of")
    
     # Embed Text Parser
    verify_embeddings_parser = subparsers.add_parser(
        "verify_embeddings", help="Verify embeddings"
    )
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            # Call the verify_model function to print the model information
            verify_model()
            return
        case "embed_text":
            embed_text(args.text)
        
        case "verify_embeddings":
            verify_embeddings()
        
        case _:
            parser.print_help()
            
        

if __name__ == "__main__":
    main()