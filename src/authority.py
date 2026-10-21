import math
from dataclasses import replace
from datetime import UTC, datetime

from src.connectors.base import ensure_utc
from src.preprocessor import RankedChunk

RECENCY_HALF_LIFE_DAYS = 365
RECENCY_FLOOR = 0.5  # an old postmortem can still be the answer, so age never more than halves a score


def recency(when: datetime, now: datetime | None = None) -> float:
    """1.0 for today, halving every RECENCY_HALF_LIFE_DAYS, never below RECENCY_FLOOR."""
    age_days = max(0.0, ((now or datetime.now(UTC)) - ensure_utc(when)).total_seconds() / 86400)
    return max(RECENCY_FLOOR, 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS))


class AuthorityScorer:
    """Final score = relevance × 1.25 for trusted (tier A) sources × recency."""

    def score(self, chunks: list[RankedChunk]) -> list[RankedChunk]:
        scored = []
        for chunk in chunks:
            # Cross-encoder scores are raw logits (often negative): squash to (0, 1) first, otherwise
            # multiplying by a boost > 1 would push a negative score further DOWN
            if chunk.cross_encoder_score is not None:
                relevance = 1.0 / (1.0 + math.exp(-chunk.cross_encoder_score))
            else:
                relevance = chunk.score  # fusion score, already positive
            tier_boost = 1.25 if chunk.metadata.authority_tier == "A" else 1.0
            scored.append(replace(chunk, score=relevance * tier_boost * recency(chunk.metadata.created_at)))
        return sorted(scored, key=lambda c: c.score, reverse=True)
