from fastapi.testclient import TestClient
from app.main import app, pipeline

client = TestClient(app)

# Tiny "golden set" = retrieval evaluation: question -> expected source file
GOLDEN = [
    ("How many days do I have to return an item?", "returns.md"),
    ("Is express shipping expensive?", "shipping.md"),
    ("When is support open?", "support.md"),
    ("How do I reset my password?", "support.md"),
    ("Can I get a refund on a gift card?", "returns.md"),
]


def test_retrieval_hit_at_1():
    hits = sum(pipeline.retrieve(q, 1)[0][0].source == src for q, src in GOLDEN)
    assert hits / len(GOLDEN) >= 0.8


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_ask_returns_sources():
    r = client.post("/ask", json={"question": "What is the return window?"})
    assert r.status_code == 200 and "returns.md" in r.json()["sources"]


def test_out_of_scope_refuses():
    r = client.post("/ask", json={"question": "Who won the cricket world cup?"})
    assert r.json()["mode"] == "refusal"


def test_validation():
    assert client.post("/ask", json={"question": "a"}).status_code == 422
