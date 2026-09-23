# Word Embedding FastAPI

This project extends the Module 3 FastAPI application by adding
a word embedding API endpoint.

## Endpoint

POST /embedding

Example request:

{
    "word": "apple"
}

The endpoint returns the word and its embedding vector.

## Build Docker Image

docker build -t sps-genai .

## Run Docker Container

docker run -p 8000:80 sps-genai

## Test the API

After starting the container, open:

http://127.0.0.1:8000/docs

Select POST /embedding and test with:

{
    "word": "apple"
}