# rag-document-qa-api

# Document Question-Answering API (RAG)

An API that answers questions strictly from documents you upload, instead of relying on an LLM's general memory. Built with FastAPI, SentenceTransformers, Qdrant, and Groq.

## How it works

**`POST /upload`**
1. Accepts a `.txt` file
2. Cleans the text and splits it into paragraph chunks
3. Converts each chunk into a 384-dimensional vector using an embedding model
4. Stores the vectors and their original text in a vector database

**`POST /ask`**
1. Converts the question into a vector
2. Finds the top 3 most similar chunks using cosine similarity
3. Sends those chunks to an LLM with the instruction: "Answer using ONLY this text"
4. Returns the answer plus the source chunks used

## Tech stack

| Component | Tool |
|---|---|
| Web framework | FastAPI |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector database | Qdrant (in-memory) |
| LLM | Groq|


## Run it locally

```bash
git clone https://github.com/wiryfountain728/rag-document-qa-api.git
cd rag-document-qa-api
pip install -r requirements.txt
cp .env.example .env   # then add your Groq API key
uvicorn [main]:app --reload
```

Open http://127.0.0.1:8000/docs to try the endpoints.

## Limitations

- Qdrant runs in memory, so uploaded documents are lost when the server restarts
- Only supports `.txt` files
- Chunking is simple paragraph splitting, so very long paragraphs may reduce answer quality

## What I learned

- [1 line: e.g., how embeddings turn text into searchable vectors]
- [1 line: e.g., why prompting the LLM to use ONLY the retrieved text reduces hallucination]
- [1 line: something that broke or surprised you]

## Future improvements

- [ ] Persistent Qdrant storage
- [ ] PDF support
- [ ] Better chunking with overlap
