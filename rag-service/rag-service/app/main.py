import logging
import time
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.rag import RagPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("rag")

app = FastAPI(title="RAG Q&A Service")
pipeline = RagPipeline(Path(__file__).parent.parent / "data")  # built once at startup


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Answer(BaseModel):
    answer: str
    mode: str
    confidence: float
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(pipeline.chunks)}


@app.post("/ask", response_model=Answer)
def ask(q: Question):
    start = time.perf_counter()
    result = pipeline.answer(q.question)
    log.info("q=%r mode=%s conf=%.2f latency_ms=%.0f", q.question, result["mode"],
             result["confidence"], (time.perf_counter() - start) * 1000)
    return result
