"""Dense embeddings from an OpenAI-compatible /embeddings endpoint (Gemini by default)."""

import asyncio

import httpx

from src.qdrant_setup import DENSE_VECTOR_SIZE
from src.settings import Settings

BATCH_SIZE = 20  # texts per request; 50 code-heavy chunks can exceed the free tier's tokens-per-minute alone
RATE_LIMIT_RETRIES = 5  # a big ingest can pass the free tier's per-minute limit: wait and retry
RATE_LIMIT_WAIT_SECONDS = 60


class ApiEmbedder:
    """
    Turns text into vectors with one HTTP call per batch. Ingest and search must use the
    same model and size, otherwise question and document vectors are not comparable.
    """

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None) -> None:
        self.url = f"{settings.llm_base_url.rstrip('/')}/embeddings"
        self.headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
        self.model = settings.embedding_model
        self.http_client = http_client

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one vector (DENSE_VECTOR_SIZE numbers) per text, in the same order."""
        vectors: list[list[float]] = []
        client = self.http_client or httpx.AsyncClient(timeout=60.0)
        try:
            for start in range(0, len(texts), BATCH_SIZE):
                for attempt in range(RATE_LIMIT_RETRIES + 1):
                    response = await client.post(self.url, headers=self.headers, timeout=60.0, json={
                        "model": self.model, "input": texts[start:start + BATCH_SIZE], "dimensions": DENSE_VECTOR_SIZE,
                    })
                    if response.status_code != 429 or attempt == RATE_LIMIT_RETRIES:
                        break
                    await asyncio.sleep(RATE_LIMIT_WAIT_SECONDS)
                response.raise_for_status()
                vectors += [item["embedding"] for item in response.json()["data"]]  # same order as the input
        finally:
            if self.http_client is None:
                await client.aclose()
        return vectors
