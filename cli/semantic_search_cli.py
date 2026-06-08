import argparse
from lib.semantic_search import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build Parser
    verify_parser = subparsers.add_parser(
        "verify", help="Verify model initialization"
    )
    
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            # Call the verify_model function to print the model information
            verify_model()
            return
        
        
        case _:
            parser.print_help()
            
        

if __name__ == "__main__":
    main()