from __future__ import annotations

from datetime import UTC, datetime

from src.connectors.base import RawDocument
from src.preprocessor import ChunkMetadata, Preprocessor, authority_tier, tag_technical_ids


def _raw_document(
    *,
    source_type: str = "runbook",
    raw_text: str = "content",
    metadata: dict[str, object] | None = None,
) -> RawDocument:
    return RawDocument(
        doc_id=f"{source_type}-doc",
        source_type=source_type,  # type: ignore[arg-type]
        source_uri=f"{source_type}://doc",
        title=f"{source_type} title",
        raw_text=raw_text,
        updated_at=datetime(2026, 4, 19, tzinfo=UTC),
        author="tester",
        metadata=metadata or {},
    )


def test_tag_technical_ids_wraps_ip_address():
    tagged = tag_technical_ids("client ip 192.168.1.1 could not connect")

    assert "<IP_ADDR:192.168.1.1>" in tagged


def test_chunk_runbook_of_1000_tokens_yields_expected_chunk_count():
    preprocessor = Preprocessor()
    doc = _raw_document(
        source_type="runbook",
        raw_text=" ".join(f"token{i}" for i in range(1000)),
    )

    chunks = preprocessor.chunk(doc)

    assert len(chunks) == 3
    assert chunks[0].chunk_id == "runbook-doc-chunk-0"
    assert chunks[-1].chunk_id == "runbook-doc-chunk-2"


def test_authority_tier_by_source_type():
    cases = [
        ("postmortem", {}, "A"),
        ("runbook", {}, "A"),
        ("doc", {}, "A"),
        ("github", {"pr_merged": True}, "A"),
        ("github", {"pr_merged": False}, "B"),
        ("jira", {}, "B"),
        ("slack", {}, "B"),
    ]

    for source_type, metadata, expected_tier in cases:
        assert authority_tier(_raw_document(source_type=source_type, metadata=metadata)) == expected_tier




def test_chunk_header_and_metadata_round_trip_through_the_payload():
    preprocessor = Preprocessor(service="payment")
    doc = RawDocument(
        doc_id="github:acme/shop/pull/12", source_type="github", source_uri="https://github.com/acme/shop/pull/12",
        title="Retry card charges (#12)", raw_text="Retry card charges (#12)\n\nCharges are retried 3 times.",
        updated_at=datetime(2026, 10, 2, 9, tzinfo=UTC),
        author="dev", metadata={"kind": "pr", "pr_merged": True},
    )

    chunk = preprocessor.chunk(doc)[0]
    # The header is embedded with the text
    assert chunk.content.splitlines()[0] == "[pr · payment · 2026-10-02]"

    metadata = chunk.metadata
    payload = metadata.to_payload(chunk.content)
    assert payload["service"] == "payment" and payload["ref"] == "12" and payload["changed_on"] == "2026-10-02"
    assert payload["created_at"] == "2026-10-02T00:00:00+00:00"  # the merge date, not the fetch date

    assert ChunkMetadata.from_payload(payload) == metadata
