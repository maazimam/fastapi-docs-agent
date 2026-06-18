"""
vectorstore.py

Embeds all corpus chunks using OpenAI embeddings and stores
them as a numpy array for similarity search.

Build the store:
    python baseline/vectorstore.py

This writes two files to baseline/:
    chunks.json      — the chunk metadata and text
    embeddings.npy   — the embedding vectors (float32)
"""
import json
import os
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from baseline.chunker import load_all_chunks

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL   = "text-embedding-3-small"
CHUNKS_PATH   = Path("baseline/chunks.json")
EMBEDDINGS_PATH = Path("baseline/embeddings.npy")
BATCH_SIZE    = 100   # embed 100 chunks per API call

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using OpenAI.
    Returns a numpy array of shape (len(texts), embedding_dim).
    """
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )
        # response.data is a list of embedding objects, one per text
        batch_embeddings = [e.embedding for e in response.data]
        all_embeddings.extend(batch_embeddings)

        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks...")
        time.sleep(0.5)   # be polite to the API

    return np.array(all_embeddings, dtype=np.float32)


def build(force: bool = False):
    """Embed all chunks and save to disk."""
    if CHUNKS_PATH.exists() and EMBEDDINGS_PATH.exists() and not force:
        print("Vector store already exists. Use force=True to rebuild.")
        return

    print("Loading chunks...")
    chunks = load_all_chunks()
    print(f"  {len(chunks)} chunks loaded.")

    print("Embedding chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    print("Saving to disk...")
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2))
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Done. {len(chunks)} chunks embedded and saved.")


def load() -> tuple[list[dict], np.ndarray]:
    """Load chunks and embeddings from disk."""
    chunks     = json.loads(CHUNKS_PATH.read_text())
    embeddings = np.load(EMBEDDINGS_PATH)
    return chunks, embeddings


def search(query: str, chunks: list[dict], embeddings: np.ndarray, top_k: int = 5) -> list[dict]:
    """
    Embed the query and return the top_k most similar chunks.
    Each result includes the chunk plus its similarity score.
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[query],
    )
    query_vec = np.array(response.data[0].embedding, dtype=np.float32)

    # cosine similarity = dot product of normalized vectors
    norms      = np.linalg.norm(embeddings, axis=1)
    normalized = embeddings / norms[:, np.newaxis]
    query_norm = query_vec / np.linalg.norm(query_vec)
    scores     = normalized @ query_norm

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        result = chunks[idx].copy()
        result["score"] = float(scores[idx])
        results.append(result)
    return results


if __name__ == "__main__":
    build()