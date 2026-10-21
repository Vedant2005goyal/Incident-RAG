import httpx
from qdrant_client import AsyncQdrantClient

from src.embeddings import ApiEmbedder
from src.preprocessor import RankedChunk
from src.schemas import KnowledgeRequest
from src.settings import Settings


class DenseRetriever:
    """Finds chunks whose meaning is close to the question (vector similarity)."""

    def __init__(self, qdrant_client: AsyncQdrantClient, http_client: httpx.AsyncClient, settings: Settings) -> None:
        self.qdrant_client = qdrant_client
        self.collection = settings.dense_collection
        self.embedder = ApiEmbedder(settings, http_client)

    async def retrieve(self, request: KnowledgeRequest) -> list[RankedChunk]:
        # Embed the question with the same model used at ingest, otherwise the vectors aren't comparable
        vectors = await self.embedder.embed_texts([request.semantic_query])
        if not vectors:
            return []
        points = await self.qdrant_client.search(
            collection_name=self.collection,
            query_vector=vectors[0],
            limit=request.max_chunks * 3,
            with_payload=True,
        )
        return [RankedChunk.from_point(point) for point in points]
