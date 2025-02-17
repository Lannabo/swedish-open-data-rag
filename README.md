# Swedish Open Data RAG

A Retrieval Augmented Generation (RAG) system for querying Swedish open data using:
- Mistral-7B for text generation
- Swedish BERT for embeddings
- FAISS for vector search
- SQLite for document storage

## Setup
1. Install requirements: `pip install -r requirements.txt`
2. Set HF_TOKEN environment variable
3. Run tests: `python test_rag.py`

## Status
Currently debugging API connection to dataportal.se
