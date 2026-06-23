# Corrective RAG Mobile Recommendation System

## Project Objective

The objective of this project is to build an intelligent Mobile Recommendation System using Corrective Retrieval-Augmented Generation (CRAG). The system retrieves relevant mobile phone information from a vector database and provides accurate recommendations based on user requirements.

## Technologies Used

* Python
* FastAPI
* Weaviate Vector Database
* Sentence Transformers
* Ollama
* Pandas
* Docker

## Project Workflow

### 1. Data Collection

A mobile phone dataset in CSV format is used. The dataset contains information such as:

* Brand
* Mobile Name
* Price
* Rating
* Storage
* Camera
* Battery
* Processor
* Operating System

### 2. Chunking

Each mobile record is converted into a text chunk containing all important specifications.

Example:

Brand: Samsung
Mobile Name: Samsung Galaxy M35
Price: 19999
Camera: 50MP
Battery: 6000mAh
Storage: 128GB

Chunking helps improve semantic retrieval and reduces irrelevant search results.

### 3. Embedding Generation

The Sentence Transformer model "all-MiniLM-L6-v2" converts each chunk into vector embeddings.

### 4. Vector Storage

The generated vectors and metadata are stored in the Weaviate Vector Database for semantic search.

### 5. Query Processing

When a user enters a query such as:

"Mobile with good camera under 50000"

the query is converted into embeddings using the same embedding model.

### 6. Retrieval

Weaviate performs vector similarity search and retrieves the most relevant mobile chunks.

### 7. Corrective RAG

A corrective validation step checks the similarity between the user query and retrieved documents.

If relevance is low:

* Additional documents are retrieved.
* Results are re-ranked.
* The best matching mobile is selected.

This reduces retrieval errors and improves recommendation quality.

### 8. Multi-Model Evaluation

The retrieved context is evaluated using multiple Ollama models:

* tinyllama
* qwen2.5:0.5b
* smollm2:360m

Each model generates a recommendation score.

### 9. Ranking

Recommendations are ranked according to their scores and the best recommendation is returned as the final output.

## API Endpoints

### POST Method

POST /recommend

Request:

{
"query": "Mobile with good camera"
}

### GET Method

GET /recommend?query=Mobile with good camera

## Output

The API returns:

* User Query
* Retrieved Chunks
* Top Recommendation
* Total Suggestions
* All Ranked Suggestions
* Model Scores

## Features

* Corrective RAG Architecture
* Semantic Search
* Chunking Support
* Weaviate Vector Database
* FastAPI Swagger UI
* GET and POST APIs
* Multi-Model Ranking
* Real-Time Mobile Recommendations

## Conclusion

This project successfully implements a Corrective Retrieval-Augmented Generation system for mobile recommendations. By combining chunking, semantic embeddings, vector search, corrective retrieval, and multi-model evaluation, the system provides accurate and context-aware recommendations based on user requirements.
