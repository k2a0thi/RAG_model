
import os
import requests
from bs4 import BeautifulSoup
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, Index
from pyngrok import ngrok
import logging
import uvicorn
import nest_asyncio

# Necessary for Colab or environments that don't support asyncio natively
nest_asyncio.apply()

# Initialize FastAPI app
app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load question-answering model (DistilBERT)
qa_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
qa_model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

# Initialize the QA pipeline
qa_pipeline = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)

# Connect to Milvus (Assuming Milvus is running locally on default port 19530)
connections.connect("default", host="localhost", port="19530")

# Define the collection schema for storing vectors
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=qa_model.config.hidden_size),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)
]

collection_name = "text_embedding_collection"

# Create or load the collection in Milvus
try:
    collection = Collection(name=collection_name)
    logger.info(f"Collection '{collection_name}' already exists.")
except Exception as e:
    schema = CollectionSchema(fields, description="Text embedding collection")
    collection = Collection(name=collection_name, schema=schema)
    logger.info(f"Collection '{collection_name}' created.")

# Create an index on the embedding field for faster search
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
    "metric_type": "L2"
}
Index(collection, "embedding", index_params)
logger.info(f"Index created on collection '{collection_name}'.")

# Load the collection for querying and inserting data
collection.load()

# Pydantic models to structure the API input
class LoadDataRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    query: str

# Web scraping function to extract data from a URL
def extract_data(url: str) -> str:
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to load page, status code: {response.status_code}")
   
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all('p')
    text = "  ".join([para.get_text() for para in paragraphs])
    return text

# Function to embed text and store it into Milvus
def embed_and_store_text(text: str):
    sentences = text.split('.')
    embeddings = qa_model(**qa_tokenizer(sentences, return_tensors="pt", padding=True)).last_hidden_state[:, 0, :].detach().numpy().tolist()
    entities = [embeddings, sentences]
    collection.insert(entities)

# Function to perform question answering using DistilBERT
def question_answer(query: str, context: str):
    return qa_pipeline(question=query, context=context)['answer']

# Endpoint to load data from a URL and store embeddings in Milvus
@app.post("/load", response_model=dict)
async def load_data(request: LoadDataRequest):
    try:
        content = extract_data(request.url)
        embed_and_store_text(content)
        return {"message": "Data loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to query Milvus and perform question-answering using DistilBERT
@app.post("/query", response_model=dict)
async def query_data(request: QueryRequest):
    try:
        if collection.num_entities == 0:
            raise HTTPException(status_code=400, detail="No data loaded")

        query_embedding = qa_model(**qa_tokenizer([request.query], return_tensors="pt", padding=True)).last_hidden_state[:, 0, :].detach().numpy().tolist()[0]
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 50}
        }
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=3,
            output_fields=["text"]
        )

        # Retrieve the most similar sentences
        best_sentences = [hit.entity.get("text") for hit in results[0]]

        # Perform question answering using the QA pipeline
        context = " ".join(best_sentences)
        answer = question_answer(request.query, context)
       
        return {
            "query": request.query,
            "best_match_sentences": best_sentences,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Start Ngrok tunnel to expose the local FastAPI app to the internet
ngrok.set_auth_token("2nT2qIlYifMZkKxCYnGKQDLjtsQ_3mXDqboctgTcC4GKKXPHV")  # Set your Ngrok auth token
ngrok_tunnel = ngrok.connect(8000)
print(f"Public URL: {ngrok_tunnel.public_url}")

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
