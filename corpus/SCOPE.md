# Project Scope

## Corpus
FastAPI documentation at commit 82064857539e6286522c347b4b11331b48dd2378 (v0.136.3).
153 pages, 433 code includes resolved. API reference extracted from source docstrings.

## What counts as in-scope
Any question a FastAPI user would reasonably expect the FastAPI docs to answer.

## Question Categories

**Single-hop answerable**
Fully answerable from one section of the corpus. Example: "What does the `status_code` parameter do in `@app.post()`?"

**Multi-hop answerable**
Requires combining two or more non-contiguous sections. Example: "How do you use OAuth2 with scopes and also return a custom response model?"

**Out-of-domain (abstain)**
Not a FastAPI usage question at all. Example: "How do I train a neural network in PyTorch?"

**In-domain-uncovered (abstain)**
A real FastAPI usage question the frozen corpus genuinely does not answer. Example: "How do I rate-limit requests per client in FastAPI?"
This is the hardest category and the most valuable one.

## Starlette / Pydantic / Uvicorn boundary rule
In-domain = anything a FastAPI user expects the FastAPI docs to answer.
Out-of-domain = pure internals of Starlette/Pydantic/Uvicorn with no FastAPI usage path.

## Decided borderline cases
1. "How does Starlette's routing work internally?" → OUT-OF-DOMAIN (pure Starlette internal)
2. "How do I add custom middleware using Starlette's BaseHTTPMiddleware?" → IN-DOMAIN-UNCOVERED (FastAPI exposes this, docs mention it but don't fully document it)
3. "How do I write a custom Pydantic validator for a request body field?" → SINGLE-HOP ANSWERABLE (FastAPI docs cover this directly)
4. "How do I configure Uvicorn workers for production?" → IN-DOMAIN-UNCOVERED (deployment question a FastAPI user asks, docs mention Uvicorn but don't document worker config)
