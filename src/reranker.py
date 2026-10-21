from dataclasses import replace

from sentence_transformers import CrossEncoder

from src.preprocessor import RankedChunk
from src.settings import Settings


class CrossEncoderReranker:
    """Re-scores chunks with a cross-encoder, which reads the question and the chunk together (slower, more accurate)."""

    def __init__(self, settings: Settings) -> None:
        self.model = CrossEncoder(settings.cross_encoder_model, max_length=512)

    def rerank(self, query: str, chunks: list[RankedChunk]) -> list[RankedChunk]:
        if not chunks:
            return []
        scores = self.model.predict([[query, chunk.content] for chunk in chunks])
        reranked = [replace(chunk, cross_encoder_score=float(score)) for chunk, score in zip(chunks, scores, strict=True)]
        return sorted(reranked, key=lambda c: c.cross_encoder_score, reverse=True)
