import re
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime

from .connectors.base import RawDocument, ensure_utc
from .documents import parse

# Technical IDs get wrapped as <TAG:value> so compression never drops them (see id_preservation.py)
ID_PATTERNS = [
    ("IP_ADDR", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), 0),
    ("TRACE_ID", re.compile(r"\b(?:trace|trace_id|trace-id|span|span_id|span-id)[=:]\s*([A-Za-z0-9-]{6,64})\b", re.IGNORECASE), 1),
    ("HASH", re.compile(r"\b[0-9a-f]{7,40}\b"), 0),
    ("PATH", re.compile(r"(?<!<)(?:/[^\s<>]+)+"), 0),
    ("ERROR_CODE", re.compile(r"\b(?:ERR|E)[A-Z0-9_]{2,}\b"), 0),
]


@dataclass
class ChunkMetadata:
    """What is stored next to a chunk's vectors in Qdrant (the payload), apart from its text."""

    chunk_id: str
    doc_id: str
    source_type: str  # postmortem, runbook, doc, github or slack
    source_uri: str  # link used in citations
    authority_tier: str  # "A" = trusted (postmortems, runbooks, docs, merged PRs), else "B"
    created_at: datetime  # the date that matters for recency: incident date, merge date, else last update
    service: str = ""
    title: str = ""
    section: str = ""  # which part of the document, e.g. "Why it happened"
    chunk_index: int = 0
    extra: dict = field(default_factory=dict)  # the type's own fields, e.g. severity, PR number, channel

    def to_payload(self, content: str) -> dict:
        """The Qdrant payload: these fields (`extra` flattened in) plus the chunk text."""
        payload = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "extra"}
        payload["created_at"] = self.created_at.isoformat()
        return {**payload, **self.extra, "content": content}

    @classmethod
    def from_payload(cls, payload: dict) -> "ChunkMetadata":
        """Rebuild the metadata of a search hit from its Qdrant payload."""
        names = {f.name for f in fields(cls)} - {"extra"}
        values = {key: value for key, value in payload.items() if key in names}
        values["created_at"] = datetime.fromisoformat(values["created_at"])
        extra = {key: value for key, value in payload.items() if key not in names and key != "content"}
        return cls(**values, extra=extra)


@dataclass
class ProcessedChunk:
    """A chunk ready to be embedded and stored."""

    chunk_id: str
    content: str
    metadata: ChunkMetadata


@dataclass
class RankedChunk:
    """A chunk found by search, with its scores."""

    chunk_id: str
    content: str
    metadata: ChunkMetadata
    score: float  # search / fusion / authority score
    cross_encoder_score: float | None = None

    @classmethod
    def from_point(cls, point) -> "RankedChunk":
        """Build from a Qdrant search hit."""
        payload = point.payload or {}
        metadata = ChunkMetadata.from_payload(payload)
        return cls(chunk_id=metadata.chunk_id, content=payload.get("content", ""), metadata=metadata, score=point.score)


class Preprocessor:
    """Turns a fetched document into chunks with metadata."""

    def __init__(self, service: str = "") -> None:
        self.service = service  # the service under test, stored on every chunk

    def chunk(self, doc: RawDocument) -> list[ProcessedChunk]:
        """Chunk the document the way its type is read (src/documents.py), with metadata on every chunk.

        Each chunk's text starts with a header such as `[postmortem · payment · 2026-10-17 · Why it happened]`,
        so the metadata is embedded together with the text. Raises pydantic.ValidationError if the
        document doesn't have its type's format.
        """
        model = parse(doc)
        day = model.content_date()
        extra = model.metadata()
        kind = extra.get("kind") or doc.source_type  # "pr" / "commit" say more than "github"
        created_at = datetime(day.year, day.month, day.day, tzinfo=UTC) if day else ensure_utc(doc.updated_at)

        chunks = []
        for index, piece in enumerate(model.chunks()):
            header = " · ".join(part for part in (kind, self.service, day.isoformat() if day else "", piece.section) if part)
            metadata = ChunkMetadata(
                chunk_id=f"{doc.doc_id}-chunk-{index}",
                doc_id=doc.doc_id,
                source_type=doc.source_type,
                source_uri=doc.source_uri,
                authority_tier=authority_tier(doc),
                created_at=created_at,
                service=self.service,
                title=doc.title,
                section=piece.section,
                chunk_index=index,
                extra=extra,
            )
            content = f"[{header}]\n{tag_technical_ids(piece.text)}"
            chunks.append(ProcessedChunk(chunk_id=metadata.chunk_id, content=content, metadata=metadata))
        return chunks


def authority_tier(doc: RawDocument) -> str:
    """"A" for postmortems, runbooks, official docs and merged PRs; "B" for the rest."""
    if doc.source_type in {"postmortem", "runbook", "doc"}:
        return "A"
    if doc.source_type == "github" and doc.metadata.get("pr_merged"):
        return "A"
    return "B"


def tag_technical_ids(text: str) -> str:
    """Wrap IPs, trace IDs, hashes, paths and error codes as <TAG:value>, e.g. <IP_ADDR:10.0.0.5>."""
    matches = []  # (start, end, tag, value), never overlapping
    for tag, pattern, group in ID_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span(group)
            if all(end <= s or start >= e for s, e, _, _ in matches):
                matches.append((start, end, tag, match.group(group)))

    result, cursor = [], 0
    for start, end, tag, value in sorted(matches):
        result.append(text[cursor:start] + f"<{tag}:{value}>")
        cursor = end
    return "".join(result) + text[cursor:]
