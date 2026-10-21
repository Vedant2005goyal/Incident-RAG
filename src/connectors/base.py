from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

SourceType = Literal["postmortem", "runbook", "doc", "github", "slack"]
MARKDOWN_TYPES = ("postmortem", "runbook", "doc")  # what a Markdown file can be


@dataclass
class RawDocument:
    """A fetched document, before it is parsed and chunked."""

    doc_id: str
    source_type: SourceType
    source_uri: str  # link used in citations
    title: str
    raw_text: str
    updated_at: datetime
    author: str
    metadata: dict = field(default_factory=dict)  # source-specific extras (front matter, PR kind, Slack messages)


class BaseConnector:
    """A source of documents. Each connector implements `fetch`."""

    async def fetch(self, since: datetime | None) -> list[RawDocument]:
        """Return the documents changed after `since` (all of them when `since` is None)."""
        raise NotImplementedError


def markdown_type(metadata: dict, default: SourceType) -> SourceType:
    """A Markdown file's type: `source_type` from its front matter if valid, else `default`."""
    declared = metadata.get("source_type")
    return declared if declared in MARKDOWN_TYPES else default


def ensure_utc(value: datetime) -> datetime:
    """Give a datetime a UTC timezone, so dates from different sources compare correctly."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
