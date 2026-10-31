from contextlib import asynccontextmanager

import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer

DOCUMENTS = [
     {
        "id": "doc-1",
        "text": "To reset your password, open account settings and select Reset Password.",
    },
    {
        "id": "doc-2",
        "text": "Deploy a Go backend service to AWS using Docker and ECS.",
    },
    {
        "id": "doc-3",
        "text": "Enable multi-factor authentication to improve account security.",
    },
    {
        "id": "doc-4",
        "text": "Configure PostgreSQL connection pooling in your backend service.",
    },
    {
        "id": "doc-5",
        "text": "If you forget your login credentials, use the account recovery page.",
    },
]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model once when the API starts
    app.state.model = SentenceTransformer(MODEL_NAME)

    # Embed all documents once
    texts = [doc["text"] for doc in DOCUMENTS]

    vectors = app.state.model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    vectors = np.asarray(vectors, dtype="float32")

    # Exact cosine-similarity search using inner product
    dimension = vectors.shape[1]
    app.state.index = faiss.IndexFlatIP(dimension)

    # FAISS returns the position of each matching document
    app.state.index.add(vectors)

    yield

    # Cleanup can be added here if needed


app = FastAPI(
    title="Semantic Search API",
    lifespan=lifespan,
)


@app.post("/search")
def search(request: SearchRequest):
    query = request.query.strip()

    if not query:
        return {"results": []}

    # Create one embedding for the incoming query
    query_vector = app.state.model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_vector = np.asarray(query_vector, dtype="float32")

    # Search for the top-k nearest document vectors
    scores, indices = app.state.index.search(
        query_vector,
        request.top_k,
    )

    results = []

    SIMILARITY_THRESHOLD = 0.50

    for score, idx in zip(scores[0], indices[0]):
        # FAISS uses -1 when no result exists
        if idx == -1:
            continue

        if score < SIMILARITY_THRESHOLD:
            continue

        results.append({
            "id": DOCUMENTS[idx]["id"],
            "text": DOCUMENTS[idx]["text"],
            "score": round(float(score), 4),
        })

    return {
        "query": query,
        "results": results,
    }
