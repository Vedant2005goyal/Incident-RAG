
from fastapi.testclient import TestClient

from src.ingest_service import IngestResponse
from src.main import app


from unittest.mock import patch

@patch("src.main.LLMLingua2Compactor")
@patch("src.main.CrossEncoderReranker")
@patch("src.main.SparseRetriever")
@patch("src.main.DenseRetriever")
def test_health_ok(mock_dense, mock_sparse, mock_reranker, mock_compactor):
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class _FakeIngestService:
    connectors = {"local_file": None}

    async def ingest(self, request):
        return IngestResponse(
            source_type=request.source_type,
            documents_indexed=0,
            chunks_indexed=0,
        )


@patch("src.main.LLMLingua2Compactor")
@patch("src.main.CrossEncoderReranker")
@patch("src.main.SparseRetriever")
@patch("src.main.DenseRetriever")
def test_ingest_endpoint_ok(mock_dense, mock_sparse, mock_reranker, mock_compactor):
    with TestClient(app) as client:
        client.app.state.ingest_service = _FakeIngestService()
        response = client.post("/ingest", json={"source_type": "local_file", "full_reindex": True})

    assert response.status_code == 200
    assert response.json() == {
        "source_type": "local_file",
        "documents_indexed": 0,
        "chunks_indexed": 0,
        "documents_skipped": 0,
    }



