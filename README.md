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


## General notes

DataportalRAG manages a Retrieval-Augmented Generation (RAG) system. This system is designed to retrieve and store data, as well as run inference using a language model. Here's a breakdown of the main functions and their roles in the two main flows:

### Flow 1: Retrieve and Store Data
#### update_knowledge_base(limit: int = 100) -> None
*Purpose: Updates the local database with new data from an external API.*
*Steps:*
1. Initialize: Start a timer and establish a database connection.
2. Fetch Data: Use fetch_datasets to retrieve datasets from the API in a paginated manner.
3. Process Data: Convert API responses into structured documents using _process_datasets.
4. Batch Processing: For each batch of documents, compute embeddings using process_batch.
5. Store Data: Insert document text and embeddings into the database.
6. Update Index: Add new embeddings to the FAISS index for efficient retrieval.
7. Log Performance: Record the operation's duration in the performance metrics table.
8. Backup: Optionally backup the database and index to S3.

#### fetch_datasets(page: int = 1, page_size: int = 100) -> Optional[Dict]
*Purpose: Fetch datasets from an external API.*
*Steps:*
1. Request Data: Send a GET request to the API with pagination parameters.
2. Handle Response: Parse the JSON response and return the result.

#### _process_datasets(api_response: Dict) -> List[Dict]
*Purpose: Convert API response data into a list of structured documents.*
*Steps:*
1. Extract Fields: For each dataset, extract relevant fields like title, description, and keywords.
2. Format Text: Combine extracted fields into a single text block for each document.

### Flow 2: Run Inference
#### _query_implementation(query_text: str, top_k: int = 3) -> str
*Purpose: Process a query to retrieve relevant documents and generate a response.*
*Steps:*
1. Compute Query Embedding: Encode the query text into an embedding.
2. Retrieve Documents: Use the FAISS index to find the top-k most similar documents.
3. Fetch Document Texts: Retrieve the text of the documents from the database.
4. Generate Response: Use the language model to generate a response based on the retrieved documents.
5. Log Performance: Record the query processing time in the performance metrics table.

#### process_batch(documents: List[Dict]) -> List[np.ndarray]
*Purpose: Compute embeddings for a batch of documents.*
*Steps:*
1. Extract Texts: Get the text from each document.
2. Compute Embeddings: Use the embedding model to encode the texts into embeddings.

### Exposed Functions
*Externally Exposed:* The main functions that are likely to be used externally are update_knowledge_base and _query_implementation (though _query_implementation is intended to be used internally, it is wrapped by a cached query method).
*Purpose: These functions allow the system to update its knowledge base with new data and to process queries to generate responses.*

### Additional Functions
*backup_to_s3(): Backs up the database and index to an S3 bucket if configured.*
*cleanup(): Cleans up resources like database connections and clears memory caches.*
These functions collectively enable the system to maintain an up-to-date knowledge base and provide intelligent responses to queries by leveraging both retrieval and generation capabilities.