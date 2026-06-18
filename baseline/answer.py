"""
answer.py

Dumb single-shot RAG baseline. No loop, no tools.

Given a question:
1. Search the vector store for top-k similar chunks
2. If the best score is below ABSTAIN_THRESHOLD, abstain
3. Otherwise pass the chunks to GPT and return the answer

This is the control. Every agent improvement gets measured against it.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from baseline.vectorstore import load, search

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ABSTAIN_THRESHOLD = 0.35   # below this similarity score, abstain
TOP_K             = 5      # number of chunks to retrieve
ANSWER_MODEL      = "gpt-4o-mini"

# load once at module level so repeated calls don't reload from disk
chunks, embeddings = load()


SYSTEM_PROMPT = """You are a helpful assistant that answers questions about FastAPI.
Answer ONLY using the provided context chunks.
If the context does not contain enough information to answer, say exactly: I don't know.
Do not use any knowledge outside the provided context."""


def answer(question: str) -> dict:
    """
    Answer a question using single-shot RAG.
    Returns {"answer": str | None, "citations": list[str]}
    """
    # step 1 — retrieve
    results = search(question, chunks, embeddings, top_k=TOP_K)
    top_score = results[0]["score"]

    # step 2 — abstain if similarity too low
    if top_score < ABSTAIN_THRESHOLD:
        return {"answer": None, "citations": []}

    # step 3 — build context from retrieved chunks
    context = "\n\n---\n\n".join(
        f"Source: {r['source']}\n\n{r['text']}"
        for r in results
    )

    # step 4 — call GPT
    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
        max_tokens=500,
    )

    answer_text = response.choices[0].message.content.strip()

    # step 5 — treat "I don't know" as abstention
    if "i don't know" in answer_text.lower():
        return {"answer": None, "citations": []}

    citations = list(dict.fromkeys(r["source"] for r in results))
    return {"answer": answer_text, "citations": citations}