import re
from dataclasses import replace

from src.preprocessor import RankedChunk
from src.settings import Settings


class WRRFFusion:
    """Merges the dense and sparse result lists with Weighted Reciprocal Rank Fusion."""

    def __init__(self, settings: Settings) -> None:
        self.k = settings.wrrf_k

    def classify_query(self, query: str) -> tuple[float, float]:
        """Return (dense weight, sparse weight): exact-word search counts more when the query has IDs."""
        has_error_code = re.search(r"\b(?:ERR|E)[A-Z0-9_]{2,}\b", query, re.IGNORECASE)
        has_trace_id = re.search(r"\b[0-9a-f]{8,}\b", query, re.IGNORECASE)
        has_path = re.search(r"(?<!<)(?:/[^\s<>]+)+", query)
        if has_error_code or has_trace_id or has_path:
            return 0.3, 0.7
        if len(query.split()) > 3 and not any(char in query for char in "{}[]_\\"):
            return 0.7, 0.3  # plain-language question
        return 0.5, 0.5

    def fuse(self, dense: list[RankedChunk], sparse: list[RankedChunk], w_dense: float, w_sparse: float) -> list[RankedChunk]:
        """Score each chunk by weight / (k + rank) in each list it appears in, summed; best first."""
        chunks = {chunk.chunk_id: chunk for chunk in dense}
        for chunk in sparse:
            chunks.setdefault(chunk.chunk_id, chunk)
        scores = dict.fromkeys(chunks, 0.0)
        for rank, chunk in enumerate(dense, start=1):
            scores[chunk.chunk_id] += w_dense / (self.k + rank)
        for rank, chunk in enumerate(sparse, start=1):
            scores[chunk.chunk_id] += w_sparse / (self.k + rank)
        fused = [replace(chunk, score=scores[chunk_id]) for chunk_id, chunk in chunks.items()]
        return sorted(fused, key=lambda c: c.score, reverse=True)
