"""Integration tests for the POST /knowledge endpoint.

Tests cover:
- Basic pipeline returning docs.
- total_tokens_after_compression is populated.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from src.id_preservation import TechnicalIDPreservationLayer
from src.main import app
from src.preprocessor import ChunkMetadata, RankedChunk


def _make_chunk(
    chunk_id: str,
    content: str,
    source_type: str = "runbook",
    authority_tier: str = "A",
    score: float = 0.9,
) -> RankedChunk:
    """Create a RankedChunk with sensible defaults for testing."""
    meta = ChunkMetadata(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        source_type=source_type,
        source_uri=f"uri://{chunk_id}",
        authority_tier=authority_tier,
        created_at=datetime.now(UTC),
    )
    return RankedChunk(
        chunk_id=chunk_id,
        content=content,
        metadata=meta,
        score=score,
    )


KNOWLEDGE_REQUEST = {
    "query_id": "q-123",
    "semantic_query": "payments database connection refused",
    "max_chunks": 12,
    "compression_budget_tokens": 2000,
}


@patch("src.main.LLMLingua2Compactor")
@patch("src.main.CrossEncoderReranker")
@patch("src.main.SparseRetriever")
@patch("src.main.DenseRetriever")
def test_knowledge_pipeline_returns_docs(mock_dense, mock_sparse, mock_reranker, mock_compactor):
    """Basic pipeline returns docs."""

    chunk = _make_chunk(
        "test-chunk",
        "payments database connection refused",
    )

    with TestClient(app) as client:
        # Replace state with mocks after lifespan has run
        client.app.state.dense_retriever = AsyncMock()
        client.app.state.dense_retriever.retrieve.return_value = [chunk]
        client.app.state.sparse_retriever = AsyncMock()
        client.app.state.sparse_retriever.retrieve.return_value = []
        client.app.state.reranker = MagicMock()
        client.app.state.reranker.rerank.side_effect = lambda q, chunks: chunks
        # Use extractive-fallback compactor (no model)
        from src.compactor import LLMLingua2Compactor as RealCompactor
        client.app.state.compactor = RealCompactor(
            model_name="__nonexistent_for_tests__"
        )
        client.app.state.id_preservation = TechnicalIDPreservationLayer()

        response = client.post("/knowledge", json=KNOWLEDGE_REQUEST)

    assert response.status_code == 200
    data = response.json()
    assert data["query_id"] == "q-123"
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["chunk_id"] == "test-chunk"
    assert data["chunks"][0]["content"] == "payments database connection refused"
    assert data["total_tokens_after_compression"] > 0


@patch("src.main.LLMLingua2Compactor")
@patch("src.main.CrossEncoderReranker")
@patch("src.main.SparseRetriever")
@patch("src.main.DenseRetriever")
def test_knowledge_pipeline_compression_populates_token_count(mock_dense, mock_sparse, mock_reranker, mock_compactor):
    """Verify total_tokens_after_compression reflects actual compression."""

    # Create chunks with enough content to test compression (punctuated for sentence splitting)
    long_content = " ".join(["This is a short sentence."] * 10)
    chunk = _make_chunk("long-chunk", long_content)

    with TestClient(app) as client:
        client.app.state.dense_retriever = AsyncMock()
        client.app.state.dense_retriever.retrieve.return_value = [chunk]
        client.app.state.sparse_retriever = AsyncMock()
        client.app.state.sparse_retriever.retrieve.return_value = []
        client.app.state.reranker = MagicMock()
        client.app.state.reranker.rerank.side_effect = lambda q, chunks: chunks
        from src.compactor import LLMLingua2Compactor as RealCompactor
        client.app.state.compactor = RealCompactor(
            model_name="__nonexistent_for_tests__"
        )
        client.app.state.id_preservation = TechnicalIDPreservationLayer()

        # Use a small budget to force compression
        request = {**KNOWLEDGE_REQUEST, "compression_budget_tokens": 20}
        response = client.post("/knowledge", json=request)

    assert response.status_code == 200
    data = response.json()
    # Token count should be populated and within budget + 5%
    assert data["total_tokens_after_compression"] > 0
    assert data["total_tokens_after_compression"] <= 22  # 20 + ~5%


@patch("src.main.LLMLingua2Compactor")
@patch("src.main.CrossEncoderReranker")
@patch("src.main.SparseRetriever")
@patch("src.main.DenseRetriever")
def test_knowledge_pipeline_returns_compacted_content(mock_dense, mock_sparse, mock_reranker, mock_compactor):
    """The response content must come from the compactor output."""

    chunk = _make_chunk(
        "verbose-chunk",
        "Verbose original text that should not be returned after compaction.",
    )

    with TestClient(app) as client:
        client.app.state.dense_retriever = AsyncMock()
        client.app.state.dense_retriever.retrieve.return_value = [chunk]
        client.app.state.sparse_retriever = AsyncMock()
        client.app.state.sparse_retriever.retrieve.return_value = []
        client.app.state.reranker = MagicMock()
        client.app.state.reranker.rerank.side_effect = lambda q, chunks: chunks
        client.app.state.compactor = MagicMock()
        client.app.state.compactor.compress.return_value = "Compacted context only."
        client.app.state.id_preservation = TechnicalIDPreservationLayer()

        response = client.post("/knowledge", json=KNOWLEDGE_REQUEST)

    assert response.status_code == 200
    data = response.json()
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["chunk_id"] == "verbose-chunk"
    assert data["chunks"][0]["content"] == "Compacted context only."


@patch("src.main.LLMLingua2Compactor")
@patch("src.main.CrossEncoderReranker")
@patch("src.main.SparseRetriever")
@patch("src.main.DenseRetriever")
def test_knowledge_pipeline_returns_reinjected_ids(mock_dense, mock_sparse, mock_reranker, mock_compactor):
    """Technical IDs stripped by compaction must be present in API content."""

    chunk = _make_chunk(
        "trace-chunk",
        "Payment failure trace <TRACE_ID:abc123def456> confirms the timeout.",
    )

    with TestClient(app) as client:
        client.app.state.dense_retriever = AsyncMock()
        client.app.state.dense_retriever.retrieve.return_value = [chunk]
        client.app.state.sparse_retriever = AsyncMock()
        client.app.state.sparse_retriever.retrieve.return_value = []
        client.app.state.reranker = MagicMock()
        client.app.state.reranker.rerank.side_effect = lambda q, chunks: chunks
        client.app.state.compactor = MagicMock()
        client.app.state.compactor.compress.return_value = (
            "Payment failure trace confirms the timeout."
        )
        client.app.state.id_preservation = TechnicalIDPreservationLayer()

        response = client.post("/knowledge", json=KNOWLEDGE_REQUEST)

    assert response.status_code == 200
    data = response.json()
    assert len(data["chunks"]) == 1
    assert "<TRACE_ID:abc123def456>" in data["chunks"][0]["content"]
    assert data["total_tokens_after_compression"] == len(
        data["chunks"][0]["content"].split()
    )


@patch("src.main.LLMLingua2Compactor")
@patch("src.main.CrossEncoderReranker")
@patch("src.main.SparseRetriever")
@patch("src.main.DenseRetriever")
def test_knowledge_pipeline_keeps_each_chunk_source(mock_dense, mock_sparse, mock_reranker, mock_compactor):
    """Each retrieved document is compressed separately and keeps its own source_uri."""

    chunks = [_make_chunk("db-runbook", "Fail over db-primary."), _make_chunk("auth-runbook", "Restart auth-service.")]

    with TestClient(app) as client:
        client.app.state.dense_retriever = AsyncMock()
        client.app.state.dense_retriever.retrieve.return_value = chunks
        client.app.state.sparse_retriever = AsyncMock()
        client.app.state.sparse_retriever.retrieve.return_value = []
        client.app.state.reranker = MagicMock()
        client.app.state.reranker.rerank.side_effect = lambda q, c: c
        client.app.state.compactor = MagicMock()
        client.app.state.compactor.compress.side_effect = lambda texts, budget_tokens: texts[0]
        client.app.state.id_preservation = TechnicalIDPreservationLayer()

        data = client.post("/knowledge", json=KNOWLEDGE_REQUEST).json()

    assert [(c["source_uri"], c["content"]) for c in data["chunks"]] == [
        ("uri://db-runbook", "Fail over db-primary."),
        ("uri://auth-runbook", "Restart auth-service."),
    ]
