# RAG Q&A Service

A small, production-shaped AI service: ask a question, get an answer grounded in your documents, with sources.

## Architecture

```
Question -> FastAPI (/ask, validated)
              |
        TF-IDF retriever (top-k paragraphs, cosine similarity)
              |
     score < threshold? --yes--> "I don't know" (guardrail)
              | no
     Claude (context-only prompt)  --or--  extractive fallback (no API key)
              |
   {answer, sources, confidence, mode}
```

## Run locally

```bash
pip install -r requirements.txt
pytest -q
uvicorn app.main:app --reload
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How long do refunds take?"}'
# optional: export ANTHROPIC_API_KEY=... to switch from extractive to LLM answers
```

## Deploy (free, ~5 minutes)

**Option A: Render**
1. Push this folder to a GitHub repo.
2. Render -> New -> Web Service -> pick the repo -> Runtime: Docker.
3. Add env var `ANTHROPIC_API_KEY` (optional). Deploy.
4. Health check path: `/health`. Docs live at `/docs`.

**Option B: Hugging Face Spaces** (Docker SDK): push the repo, set `PORT=7860`.

**Option C: Cloud Run**: `gcloud run deploy rag-service --source . --allow-unauthenticated`

CI (GitHub Actions) runs tests and a Docker build on every push.

## Interview talking points

**What it is (30 sec):** "A retrieval-augmented generation service. Documents are chunked and indexed; at query time I retrieve the most relevant chunks and either have an LLM answer strictly from them or return the top chunk. It's containerized, tested, and CI-checked."

**Design decisions you can defend**
- *Why RAG, not fine-tuning?* Knowledge changes often; RAG updates by editing docs, gives citations, and is cheaper.
- *Why TF-IDF?* Dependency-light baseline, fast, deterministic, easy to debug. Honest limitation: no semantic matching ("cost" vs "price"). Upgrade path: embeddings (e.g. sentence-transformers) + a vector store (FAISS/pgvector), possibly hybrid with BM25.
- *Why paragraph chunking?* Each paragraph is a self-contained fact. For longer docs I'd use overlapping fixed-size chunks.
- *Hallucination control:* similarity threshold -> refusal; prompt restricts answers to context; sources returned so users can verify.
- *Graceful degradation:* no API key or LLM outage -> extractive fallback; service stays up.

**Engineering practices**
- Pydantic validation (422 on bad input), typed response model.
- Index built once at startup, not per request.
- Structured logging of mode, confidence, latency per request.
- Golden-set retrieval eval in tests (hit@1 >= 0.8) acts as a regression guard when you change chunking or the retriever.
- Pinned dependencies, slim Docker image, secrets via env vars, CI pipeline.

**"What would you improve?"** (have this ready)
1. Embeddings + vector DB; hybrid search; reranker.
2. Larger eval set with answer-quality metrics (faithfulness, LLM-as-judge).
3. Streaming responses, caching, rate limiting, auth.
4. Observability: tracing, token/cost tracking, drift monitoring on confidence scores.
5. Ingestion pipeline for PDFs/HTML and incremental re-indexing.

**Likely questions**
- *How do you evaluate it?* Retrieval: hit@k on a golden set. Generation: faithfulness to context + human spot checks.
- *How would it scale?* Stateless app -> horizontal scaling; move index to a vector DB; async LLM calls; cache frequent questions.
- *How do you handle prompt injection in documents?* Treat retrieved text as data, restrict output to context, sanitize/ingest-vet sources, limit what the LLM can do (no tools here).
