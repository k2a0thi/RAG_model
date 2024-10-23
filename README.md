

## Overview
This project is a FastAPI-based web application that performs the following tasks:
1. Scrapes text data from a given URL.
2. Uses a pre-trained DistilBERT model for embedding text and storing the embeddings in a Milvus vector database.
3. Provides a question-answering system using the stored embeddings and the DistilBERT model.
4. Exposes two API endpoints for loading data and querying the model.

## Requirements

### Python Libraries:
- `requests`: To scrape data from web pages.
- `BeautifulSoup4`: To parse HTML and extract text data.
- `transformers`: To use the DistilBERT model for question answering.
- `sentence_transformers`: To handle text embeddings (used within the QA model).
- `fastapi`: To create the API endpoints.
- `pydantic`: To define data models for API requests.
- `pymilvus`: To interact with Milvus, the vector database.
- `pyngrok`: To expose the API to the internet through an Ngrok tunnel.
- `uvicorn`: To serve the FastAPI application.
- `nest_asyncio`: To allow asynchronous functions in environments like Google Colab.

### External Services:
- **Milvus**: A vector database used to store and retrieve text embeddings.
- **Ngrok**: For exposing the FastAPI app on a public URL.

## Setup

### Step 1: Install Dependencies
Install the required Python libraries using pip:

\`\`\`bash
pip install requests beautifulsoup4 transformers sentence-transformers fastapi uvicorn pydantic pymilvus pyngrok nest_asyncio
\`\`\`

### Step 2: Milvus Setup
Ensure Milvus is installed and running on your local machine or server. By default, the application connects to Milvus on \`localhost:19530\`. Modify the connection details if Milvus is hosted elsewhere.

You can install Milvus locally by following the instructions at the [Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md).

### Step 3: Run the FastAPI App
1. Make sure Milvus is running on your local machine or server.
2. Set your Ngrok auth token in the code to enable internet exposure.

Run the FastAPI application:
\`\`\`bash
python app.py
\`\`\`

Ngrok will provide a public URL, and the FastAPI application will be accessible via that URL.

## Usage

### API Endpoints

#### 1. Load Data from URL
**Endpoint**: \`/load\`  
**Method**: \`POST\`  
**Description**: This endpoint accepts a URL, scrapes text data from it, embeds the text using DistilBERT, and stores the embeddings in the Milvus vector database.

**Request Body**:
\`\`\`json
{
  "url": "https://example.com"
}
\`\`\`

**Response**:
\`\`\`json
{
  "message": "Data loaded successfully"
}
\`\`\`

#### 2. Query for Question Answering
**Endpoint**: \`/query\`  
**Method**: \`POST\`  
**Description**: This endpoint accepts a query and searches for the most relevant text in the Milvus database. It uses the retrieved text to perform question answering.

**Request Body**:
\`\`\`json
{
  "query": "What is the capital of France?"
}
\`\`\`

**Response**:
\`\`\`json
{
  "query": "What is the capital of France?",
  "best_match_sentences": ["France is a country in Europe. The capital of France is Paris."],
  "answer": "Paris"
}
\`\`\`

### Example Walkthrough
1. **Loading Data**:
   Send a POST request to \`/load\` with a URL. The app will scrape text data from the webpage and store it as vector embeddings in Milvus.

2. **Querying**:
   After loading data, you can send a POST request to \`/query\` with a question. The app will search the database for relevant text and use DistilBERT to answer the question based on the best matches.

## Customization

- **Embedding Model**: The code uses DistilBERT for both embedding text and question answering. You can replace this with other models from the \`transformers\` library if desired.
- **Index Type**: The app uses \`IVF_FLAT\` for indexing embeddings in Milvus. You can experiment with other index types like \`HNSW\` or \`ANNOY\` for performance optimizations.

## Notes
- Make sure you have a reliable internet connection for Ngrok and Milvus.
- The application is designed to work in Google Colab as well, with support for asynchronous requests using \`nest_asyncio\`.

## License
This project is open-source. You are free to modify and distribute the code as needed.
