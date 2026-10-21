from qdrant_client import AsyncQdrantClient, models

from src.preprocessor import RankedChunk
from src.qdrant_setup import SPARSE_VECTOR_NAME, sparse_vector
from src.schemas import KnowledgeRequest
from src.settings import Settings


class SparseRetriever:
    """Finds chunks that contain the question's exact words (error codes, host names)."""

    def __init__(self, qdrant_client: AsyncQdrantClient, settings: Settings) -> None:
        self.qdrant_client = qdrant_client
        self.collection = settings.sparse_collection

    async def retrieve(self, request: KnowledgeRequest) -> list[RankedChunk]:
        vector = sparse_vector(request.semantic_query)
        if not vector.indices:
            return []  # no words to match
        points = await self.qdrant_client.search(
            collection_name=self.collection,
            query_vector=models.NamedSparseVector(name=SPARSE_VECTOR_NAME, vector=vector),
            limit=request.max_chunks * 3,
            with_payload=True,
        )
        return [RankedChunk.from_point(point) for point in points]
