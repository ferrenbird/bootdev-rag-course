# Welcome!
This repository contains my work for the "Learn Retrieval Augmented Generation" course on Boot.Dev

As is the case with all of my Boot.Dev, all code is my own. While LLMs (and Boots!) were used in the troubleshooting phase, all code was written by myself.

https://www.boot.dev/courses/learn-retrieval-augmented-generation


# Some references for myself

## The CLI
The following commands are currently supported:

|Flag|Description|Required Args|Examples|
|-----|-----|-----|-----|
|-build|Parses the incoming .json file and builds pickle (.pkl) files for doc_lengths, docmap, index, and term_frequencies|None|`uv run cli/keyword_search_cli.py build`|
|-search|Performs a basic search (for now - will be our main BM25 search soon though!)|<li>query (str) - term to search for</li></ul>|`uv run cli/keyword_search_cli.py 'pepper'`|
|-tf|Retrieves frequency of term in given doc_id|<li>doc_id (int) - numberical ID of doc to search</li><li>term (str) - term to search for</li>|`uv run cli/keyword_search_cli.py tf 1 'pepper'`|
|-idf|Calculate Inverse Document frequency for a given term|<li>term (str) - term to search for</li>|`uv run cli/keyword_search_cli.py idf 1 'pepper'`|

## The InvertedIndex class
The InvertedIndex class contains information regarding term frequency within a subset of movie synopsis data (from [the following data source](https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json)).

### .index
Example of index:
```
{
    'matrix': {1, 5, 10},
    'hacker': {1, 8},
    'reality': {1, 3, 7}
}
```

### .docmap
Example of docmap
```
{
    4150: {
        "id": 4150,
        "title": "Memory Lane",
        "description": "NICK BOXER has just returned from war..."
    },
    4151: {
        "id": 4151,
        "title": "Freaks of Nature",
        "description": "In an all out war who will win..."
    },
    4152: {
        "id": 4152,
        "title": "Mies vailla menneisyytt\\u00e4",
        "description": "As a non Finnish speaker I watched this with Finnish Subtitles to Finnish dialogue. This therefore is my limited understanding of the movie..."
    }
}
```

### .stopwords
List of stopwords loaded from the data/stopwords.txt file.

Used to filter out filler words from our datasets.

### .movies
Copy of movies.json file, converted to dictionary format.

### .term_frequencies

### .doc_lengths

### .doc_lengths_paths


(WIP)