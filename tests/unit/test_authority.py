from datetime import datetime, timedelta, timezone

import pytest

from src.authority import RECENCY_FLOOR, AuthorityScorer, recency
from src.preprocessor import ChunkMetadata, RankedChunk

NOW = datetime.now(timezone.utc)


def _mock_metadata(tier: str, age_days: int = 0) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id="",
        doc_id="",
        source_type="slack",
        source_uri="",
        authority_tier=tier,
        created_at=NOW - timedelta(days=age_days),
    )


def test_authority_scoring():
    scorer = AuthorityScorer()
    base_score = 10.0

    chunk_b = RankedChunk(chunk_id="tier-b", content="", metadata=_mock_metadata("B"), score=base_score)
    chunk_a = RankedChunk(chunk_id="tier-a", content="", metadata=_mock_metadata("A"), score=base_score)
    chunk_old_a = RankedChunk(chunk_id="old-tier-a", content="", metadata=_mock_metadata("A", age_days=365),
                              score=base_score)

    scored = scorer.score([chunk_old_a, chunk_b, chunk_a])

    # tier-a: 10 * 1.25 = 12.5; tier-b: 10 * 1.0 = 10; old-tier-a: 10 * 1.25 * 0.5 (a year old) = 6.25
    assert [c.chunk_id for c in scored] == ["tier-a", "tier-b", "old-tier-a"]
    assert scored[0].score == pytest.approx(12.5, rel=1e-4)
    assert scored[1].score == pytest.approx(10.0, rel=1e-4)
    assert scored[2].score == pytest.approx(6.25, rel=1e-3)


def test_recency_halves_each_year_and_has_a_floor():
    assert recency(NOW, NOW) == 1.0
    assert recency(NOW - timedelta(days=365), NOW) == pytest.approx(0.5)
    assert recency(NOW - timedelta(days=5 * 365), NOW) == RECENCY_FLOOR
    assert recency(NOW + timedelta(days=3), NOW) == 1.0  # future dates count as today


def test_boost_still_helps_when_cross_encoder_score_is_negative():
    """Same negative cross-encoder logit: the tier-A doc must rank ABOVE the tier-B doc."""
    tier_a = RankedChunk(chunk_id="a", content="", metadata=_mock_metadata("A"),
                         score=0.01, cross_encoder_score=-8.0)
    tier_b = RankedChunk(chunk_id="b", content="", metadata=_mock_metadata("B"),
                         score=0.01, cross_encoder_score=-8.0)

    scored = AuthorityScorer().score([tier_b, tier_a])

    assert [c.chunk_id for c in scored] == ["a", "b"]
    assert all(0 < c.score < 1 for c in scored)
