"""Minimal RAG pipeline: chunk -> index (TF-IDF) -> retrieve -> generate."""
import os
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MIN_SCORE = float(os.getenv("MIN_SCORE", "0.10"))  # below this we refuse to answer
MODEL = os.getenv("MODEL", "claude-sonnet-4-6")


@dataclass
class Chunk:
    text: str
    source: str


def load_chunks(data_dir: Path) -> list[Chunk]:
    """Chunk by paragraph; each paragraph is small enough to be self-contained."""
    chunks = []
    for f in sorted(data_dir.glob("*.md")):
        for para in f.read_text().split("\n\n"):
            para = para.strip()
            if para and not para.startswith("#"):
                chunks.append(Chunk(para, f.name))
    return chunks


class RagPipeline:
    def __init__(self, data_dir: Path):
        self.chunks = load_chunks(data_dir)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def retrieve(self, question: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = self.vectorizer.transform([question])
        scores = cosine_similarity(q, self.matrix)[0]
        top = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def answer(self, question: str, k: int = 3) -> dict:
        hits = self.retrieve(question, k)
        best = hits[0][1]
        if best < MIN_SCORE:  # guardrail: don't hallucinate when nothing relevant
            return {"answer": "I don't know based on the available documents.",
                    "mode": "refusal", "confidence": best, "sources": []}

        context = "\n".join(f"[{c.source}] {c.text}" for c, _ in hits)
        sources = sorted({c.source for c, _ in hits})

        if os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=MODEL, max_tokens=300,
                system="Answer ONLY using the provided context. If the context "
                       "does not contain the answer, say you don't know. Be concise.",
                messages=[{"role": "user",
                           "content": f"Context:\n{context}\n\nQuestion: {question}"}],
            )
            return {"answer": msg.content[0].text, "mode": "llm",
                    "confidence": best, "sources": sources}

        # Fallback: extractive answer (top chunk) so the service works offline
        return {"answer": hits[0][0].text, "mode": "extractive",
                "confidence": best, "sources": sources}
