import asyncio
import logging
from datetime import datetime
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ValidationError
from qdrant_client import QdrantClient, models

from .connectors.base import BaseConnector, ensure_utc
from .embeddings import ApiEmbedder
from .preprocessor import Preprocessor
from .qdrant_setup import SPARSE_VECTOR_NAME, ensure_qdrant_collections, sparse_vector

logger = logging.getLogger(__name__)


class IngestRequest(BaseModel):
    source_type: str  # github or slack
    full_reindex: bool = False


class IngestResponse(BaseModel):
    source_type: str
    documents_indexed: int
    chunks_indexed: int
    documents_skipped: int = 0


class IngestService:
    """Fetch documents, chunk them, embed them and store them in Qdrant."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        connectors: dict[str, BaseConnector],
        preprocessor: Preprocessor,
        embedder: ApiEmbedder,
        dense_collection: str,
        sparse_collection: str,
    ) -> None:
        self.qdrant_client = qdrant_client
        self.connectors = connectors
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.dense_collection = dense_collection
        self.sparse_collection = sparse_collection
        self.last_indexed_at: dict[str, datetime] = {}

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        ensure_qdrant_collections(
            self.qdrant_client, self.dense_collection, self.sparse_collection
        )

        # Connectors use synchronous requests, so run fetching in a worker thread.
        since = None if request.full_reindex else self.last_indexed_at.get(request.source_type)
        documents = await asyncio.to_thread(
            self.connectors[request.source_type].fetch, since
        )

        chunks = []
        skipped = 0
        for document in documents:
            try:
                chunks.extend(self.preprocessor.chunk(document))
            except ValidationError as exc:
                skipped += 1
                logger.warning(
                    "skipped %s: not a valid %s: %s",
                    document.doc_id,
                    document.source_type,
                    exc,
                )

        if chunks:
            vectors = await self.embedder.embed_texts([chunk.content for chunk in chunks])
            dense_points = []
            sparse_points = []

            for chunk, vector in zip(chunks, vectors, strict=True):
                point_id = str(uuid5(NAMESPACE_URL, chunk.chunk_id))
                payload = chunk.metadata.to_payload(chunk.content)
                dense_points.append(
                    models.PointStruct(id=point_id, vector=vector, payload=payload)
                )
                sparse_points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={SPARSE_VECTOR_NAME: sparse_vector(chunk.content)},
                        payload=payload,
                    )
                )

            self.qdrant_client.upsert(
                collection_name=self.dense_collection, points=dense_points, wait=True
            )
            self.qdrant_client.upsert(
                collection_name=self.sparse_collection, points=sparse_points, wait=True
            )

        if documents:
            self.last_indexed_at[request.source_type] = max(
                ensure_utc(document.updated_at) for document in documents
            )

        return IngestResponse(
            source_type=request.source_type,
            documents_indexed=len(documents) - skipped,
            chunks_indexed=len(chunks),
            documents_skipped=skipped,
        )

