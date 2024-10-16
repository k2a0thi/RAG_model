# RAG_model
# Wikipedia Scraper and Question-Answering System

This project is a web scraping and question-answering system that scrapes Wikipedia pages, stores the scraped content in an SQLite database, and allows users to query the data using a question-answering model.

## Features

- Scrape content from any Wikipedia page.
- Store scraped data in an SQLite database.
- Use a question-answering (QA) model from Hugging Face’s Transformers library to answer questions based on the scraped Wikipedia data.
- Serve the app through a FastAPI server and expose it via `ngrok` for remote access.

## Project Structure

- **scrape_wikipedia**: Scrapes content from a given Wikipedia URL.
- **setup_database**: Creates an SQLite database to store scraped Wikipedia content.
- **insert_data_into_db**: Inserts the scraped data into the SQLite database.
- **get_content_from_db**: Retrieves stored Wikipedia content from the database.
- **FastAPI server**: Provides REST endpoints to load Wikipedia data.
- **Question-answering model**: Uses a Hugging Face Transformers QA model to answer questions based on the stored data.

## Requirements

Before running the project, make sure you have the following libraries installed:

- `requests`
- `beautifulsoup4`
- `sqlite3`
- `transformers`
- `FastAPI`
- `uvicorn`
- `pyngrok`
- `nest_asyncio`
- `pydantic`

You can install them by running the following command:

```bash
!pip install requests beautifulsoup4 sqlite3 transformers fastapi uvicorn pyngrok nest_asyncio pydantic
