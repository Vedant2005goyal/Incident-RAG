import asyncio
from unittest.mock import AsyncMock, MagicMock

from qdrant_client.http.models import ScoredPoint

from src.schemas import KnowledgeRequest
from src.dense import DenseRetriever
from src.settings import Settings


def test_retrieve_dense():
    # Setup test data
    settings = Settings(
        llm_base_url="http://mock-llm/v1",
        llm_api_key="k",
        dense_collection="test_dense"
    )
    
    request = KnowledgeRequest(
        query_id="q-1",
        semantic_query="test query",
        max_chunks=4,
        compression_budget_tokens=1000
    )

    # Mock HTTP client
    mock_http_response = MagicMock()
    mock_http_response.raise_for_status = MagicMock()
    mock_http_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    
    mock_http_client = AsyncMock()
    mock_http_client.post.return_value = mock_http_response

    # Mock Qdrant client
    mock_qdrant_client = AsyncMock()
    mock_point = ScoredPoint(
        id="point-1",
        version=1,
        score=0.95,
        payload={
            "chunk_id": "test-chunk-1",
            "doc_id": "doc-1",
            "content": "This is a test document.",
            "source_type": "runbook",
            "source_uri": "runbook.md",
            "created_at": "2026-05-01T10:00:00+00:00",
            "authority_tier": "A"
        }
    )
    mock_qdrant_client.search.return_value = [mock_point]

    # Initialize retriever
    retriever = DenseRetriever(
        qdrant_client=mock_qdrant_client,
        http_client=mock_http_client,
        settings=settings
    )

    # Execute
    chunks = asyncio.run(retriever.retrieve(request))

    # Assertions
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "test-chunk-1"
    assert chunks[0].score == 0.95
    assert chunks[0].content == "This is a test document."
    
    # Verify HTTP call
    mock_http_client.post.assert_called_once_with(
        "http://mock-llm/v1/embeddings",
        headers={"Authorization": "Bearer k"},
        timeout=60.0,
        json={"model": settings.embedding_model, "input": ["test query"], "dimensions": 768},
    )
    
    # Verify Qdrant search call
    mock_qdrant_client.search.assert_called_once()
    call_kwargs = mock_qdrant_client.search.call_args.kwargs
    assert call_kwargs["collection_name"] == settings.dense_collection
    assert call_kwargs["query_vector"] == [0.1, 0.2, 0.3]
    assert call_kwargs["limit"] == 12  # max_chunks (4) * 3
