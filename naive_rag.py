import os
import uuid
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 1. Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from environment or .env file.")

# 2. Initialize FastAPI
app = FastAPI(title="Naive RAG Pipeline with Groq")

# 3. Initialize Embedding Model & Qdrant Store
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Vector dimension: 384
qdrant = QdrantClient(":memory:")
COLLECTION_NAME = "documents"

# Create collection (recreate to ensure clean state on boot)
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 4. Initialize Groq via OpenAI-compatible SDK
llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

# Schema for /ask endpoint
class QueryRequest(BaseModel):
    question: str


@app.post("/upload")
#@app.get("/models")
#async def list_models():
    #models = llm_client.models.list()
    #return {"available_models": [m.id for m in models.data]}

async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")
    
    contents = await file.read()
    text = contents.decode("utf-8")
    
    # Split text into non-empty chunks
    # Standardize line endings before splitting
    clean_text = text.replace("\r\n", "\n")
    chunks = [chunk.strip() for chunk in clean_text.split("\n\n") if chunk.strip()]
    if not chunks:
        raise HTTPException(status_code=400, detail="File is empty")
    
    points = []
    for chunk in chunks:
        vector = embedding_model.encode(chunk).tolist()
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk}
            )
        )
    
    # Store in Qdrant
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    
    return {
        "status": "success",
        "message": f"Successfully indexed {len(chunks)} chunks from {file.filename}."
    }


@app.post("/ask")
async def ask_question(payload: QueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # 1. Vectorize query
    query_vector = embedding_model.encode(payload.question).tolist()
    
    # 2. Retrieve top-3 chunks (handles both legacy and modern qdrant-client syntax)
    try:
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3
        )
        points = response.points
    except AttributeError:
        points = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3
        )
    
    if not points:
        return {
            "answer": "No documents found. Please upload a file first using /upload.",
            "source_chunks": []
        }
    
    source_chunks = [pt.payload["text"] for pt in points]
    context = "\n---\n".join(source_chunks)
    
    # 3. Grounded RAG Prompting
    system_prompt = (
        "You are an assistant answering questions strictly based on the provided document context.\n"
        "If the answer cannot be determined from the context, respond strictly with: "
        "'I don't know based on the provided document.'\n"
        "Do NOT invent or assume facts outside the provided text."
    )
    
    # 4. Generate response via Groq
    try:
        completion = llm_client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {payload.question}"}
            ],
            temperature=0.0
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API Error: {str(e)}")
    
    return {
        "answer": answer,
        "source_chunks": source_chunks
    }