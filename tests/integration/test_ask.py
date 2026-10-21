"""POST /ask: the RAG pipeline result goes to the answer step and comes back with sources."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.answer import Answerer
from src.main import app
from src.schemas import KnowledgeChunk, KnowledgeResult


@patch("src.main.LLMLingua2Compactor")
@patch("src.main.CrossEncoderReranker")
@patch("src.main.SparseRetriever")
@patch("src.main.DenseRetriever")
def test_ask_returns_answer_with_sources(*_mocks):
    chunk = KnowledgeChunk(chunk_id="rb-db-failover-chunk-0", source_type="runbook",
                           source_uri="confluence://runbooks/db-primary-failover", authority_tier="A",
                           recency_score=1.0, content="Promote db-replica-1.", retrieved_at=datetime.now(UTC))
    result = KnowledgeResult(query_id="q", chunks=[chunk], total_tokens_after_compression=3)

    with TestClient(app) as client:
        client.app.state.answerer = Answerer()  # no LLM: template answer
        with patch("src.main.retrieve", AsyncMock(return_value=result)) as retrieve:
            response = client.post("/ask", json={"question": "why did payments fail?", "max_chunks": 3})

    assert response.status_code == 200
    body = response.json()
    assert retrieve.call_args.args[0].semantic_query == "why did payments fail?"
    assert retrieve.call_args.args[0].max_chunks == 3
    assert body["sources"] == [{"n": 1, "source_uri": "confluence://runbooks/db-primary-failover",
                                "chunk_id": "rb-db-failover-chunk-0"}]
    assert "db-replica-1" in body["answer"]


def test_ask_rejects_empty_question():
    with patch("src.main.CrossEncoderReranker"), patch("src.main.LLMLingua2Compactor"), TestClient(app) as client:
        assert client.post("/ask", json={"question": ""}).status_code == 422
