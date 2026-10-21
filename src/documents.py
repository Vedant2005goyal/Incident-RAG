"""One Pydantic model per document type, each chunked the way that type is read.

Connectors fetch raw text; ``parse()`` turns it into the model for its ``source_type``. Validation
rejects a document that doesn't have the expected format (e.g. a postmortem without a root cause),
and ``model.chunks()`` splits it:

| Type         | Model         | Chunks                                                                     |
|--------------|---------------|----------------------------------------------------------------------------|
| ``postmortem`` | ``Postmortem``  | 4 by question: what happened, timeline, why, how it was fixed              |
| ``runbook``    | ``Runbook``     | one per section, so a step is never cut in half                            |
| ``doc``        | ``TechDoc``     | sections packed together up to the size limit, each labelled with its heading |
| ``github``     | ``CodeChange``  | one: title + description of the PR or commit                               |
| ``slack``      | ``SlackThread`` | windows of 10 messages that overlap by 2                                   |

Every chunk starts with the document's title so it can be understood (and found) on its own, and knows
which ``section`` of the document it is. ``model.metadata()`` gives the type's own fields (incident date,
PR number, channel, ...) and ``model.content_date()`` the date that matters for recency; the preprocessor
stores both on every chunk.
"""

import re
from datetime import date
from typing import Literal, Union

from pydantic import BaseModel, Field

from .connectors.base import RawDocument

CHUNK_WORDS = 512  # longer text is split into windows of this many words
OVERLAP_WORDS = 64
SLACK_WINDOW, SLACK_OVERLAP = 10, 2  # messages per chunk, messages shared with the next chunk
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


class Section(BaseModel):
    """A Markdown heading and the text under it (``heading`` is empty for text before the first heading)."""

    heading: str
    body: str


def split_sections(markdown: str) -> list[Section]:
    """Split Markdown into sections at every heading; empty sections are dropped."""

    sections: list[Section] = []
    heading, lines = "", []
    for line in markdown.splitlines():
        match = _HEADING.match(line)
        if match:
            sections.append(Section(heading=heading, body="\n".join(lines).strip()))
            heading, lines = match.group(2), []
        else:
            lines.append(line)
    sections.append(Section(heading=heading, body="\n".join(lines).strip()))
    return [section for section in sections if section.body]


def windows(text: str, words: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Return ``text`` as-is if it is short, else overlapping windows of ``words`` words."""

    tokens = text.split()
    if len(tokens) <= words:
        return [text] if tokens else []
    parts: list[str] = []
    for start in range(0, len(tokens), words - overlap):
        parts.append(" ".join(tokens[start:start + words]))
        if start + words >= len(tokens):
            break
    return parts


class Chunk(BaseModel):
    """One piece of a document: which part it is (``section``, empty for the whole document) and its text."""

    section: str
    text: str


def _titled(title: str, section: str, body: str) -> list[Chunk]:
    """Prefix each window of ``body`` with ``title — section`` so every chunk names its document."""

    header = f"{title} — {section}" if section else title
    return [Chunk(section=section, text=f"{header}\n\n{part}") for part in windows(body)]


def _present(**fields: object) -> dict[str, object]:
    """Drop empty fields, and store dates as ISO strings (Qdrant payloads are JSON)."""

    return {key: value.isoformat() if isinstance(value, date) else value for key, value in fields.items() if value}


# Postmortem section heading (first word, lower case) → Postmortem field
POSTMORTEM_SECTIONS = {"summary": "summary", "impact": "impact", "timeline": "timeline", "root": "root_cause",
                       "trigger": "trigger", "detection": "detection", "resolution": "resolution", "lessons": "lessons"}


class Postmortem(BaseModel):
    """An incident record in the format of ``data/docs/postmortems/`` (see README)."""

    title: str = Field(min_length=1)
    service: str = ""
    incident_date: date | None = None
    severity: str = ""
    summary: str = Field(min_length=1)
    impact: str = ""
    timeline: str = ""
    root_cause: str = Field(min_length=1)
    trigger: str = ""
    detection: str = ""
    resolution: str = Field(min_length=1)
    lessons: str = ""

    @classmethod
    def from_markdown(cls, title: str, metadata: dict[str, object], markdown: str) -> "Postmortem":
        fields: dict[str, object] = {"title": title, "service": str(metadata.get("service") or ""),
                                     "incident_date": metadata.get("incident_date"),
                                     "severity": str(metadata.get("severity") or "")}
        for section in split_sections(markdown):
            words = section.heading.lower().split()
            if words and words[0] in POSTMORTEM_SECTIONS:
                fields[POSTMORTEM_SECTIONS[words[0]]] = section.body
        return cls(**fields)

    def metadata(self) -> dict[str, object]:
        return _present(incident_date=self.incident_date, severity=self.severity)

    def content_date(self) -> date | None:
        return self.incident_date

    def chunks(self) -> list[Chunk]:
        # Grouped by the question an on-call engineer asks, so "why did X fail?" lands on the cause
        groups = [
            ("What happened", [("Summary", self.summary), ("Impact", self.impact)]),
            ("Timeline", [("Timeline", self.timeline)]),
            ("Why it happened", [("Root cause", self.root_cause), ("Trigger", self.trigger),
                                 ("Detection", self.detection)]),
            ("How it was fixed", [("Resolution", self.resolution), ("Lessons and action items", self.lessons)]),
        ]
        chunks: list[Chunk] = []
        for label, parts in groups:
            body = "\n\n".join(f"{name}:\n{text}" for name, text in parts if text)
            if body:
                chunks += _titled(self.title, label, body)
        return chunks


class Runbook(BaseModel):
    """Steps for an alert or failure; each section (e.g. Meaning, Diagnosis, Mitigation) is one chunk."""

    title: str = Field(min_length=1)
    sections: list[Section] = Field(min_length=1)

    def metadata(self) -> dict[str, object]:
        return {}

    def content_date(self) -> date | None:
        return None

    def chunks(self) -> list[Chunk]:
        return [chunk for section in self.sections for chunk in _titled(self.title, section.heading, section.body)]


class TechDoc(BaseModel):
    """Technical documentation; its many short sections are packed together up to ``CHUNK_WORDS``."""

    title: str = Field(min_length=1)
    sections: list[Section] = Field(min_length=1)

    def metadata(self) -> dict[str, object]:
        return {}

    def content_date(self) -> date | None:
        return None

    def chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        current: list[Section] = []

        def flush() -> None:
            # The chunk's section lists the headings packed into it, e.g. "Traces, Metrics"
            body = "\n\n".join(f"## {s.heading}\n{s.body}" if s.heading else s.body for s in current)
            for chunk in _titled(self.title, "", body):
                chunks.append(Chunk(section=", ".join(s.heading for s in current if s.heading), text=chunk.text))

        for section in self.sections:
            words = sum(len(f"{s.heading} {s.body}".split()) for s in current + [section])
            if current and words > CHUNK_WORDS:
                flush()
                current = []
            current.append(section)
        if current:
            flush()
        return chunks


class CodeChange(BaseModel):
    """A merged PR or a commit of the service: what changed and why."""

    kind: Literal["pr", "commit"]
    ref: str = Field(min_length=1)  # PR number or commit SHA
    title: str = Field(min_length=1)
    body: str = ""
    author: str = ""
    changed_on: date | None = None  # merge date of the PR, or commit date

    def metadata(self) -> dict[str, object]:
        return _present(kind=self.kind, ref=self.ref, author=self.author, changed_on=self.changed_on)

    def content_date(self) -> date | None:
        return self.changed_on

    def chunks(self) -> list[Chunk]:
        name = f"PR #{self.ref}" if self.kind == "pr" else f"Commit {self.ref[:12]}"
        return _titled(f"{name}: {self.title}", "", self.body or self.title)


class SlackMessage(BaseModel):
    user: str
    text: str


class SlackThread(BaseModel):
    """An engineering discussion; overlapping windows keep a question next to its answer."""

    channel: str = ""
    messages: list[SlackMessage] = Field(min_length=1)
    last_message_on: date | None = None

    def metadata(self) -> dict[str, object]:
        return _present(channel=self.channel, message_count=len(self.messages), last_message_on=self.last_message_on)

    def content_date(self) -> date | None:
        return self.last_message_on

    def chunks(self) -> list[Chunk]:
        title = f"Slack thread in #{self.channel}" if self.channel else "Slack thread"
        chunks: list[Chunk] = []
        for start in range(0, len(self.messages), SLACK_WINDOW - SLACK_OVERLAP):
            window = self.messages[start:start + SLACK_WINDOW]
            chunks += _titled(title, "", "\n".join(f"{m.user}: {m.text}" for m in window))
            if start + SLACK_WINDOW >= len(self.messages):
                break
        return chunks


Document = Union[Postmortem, Runbook, TechDoc, CodeChange, SlackThread]


def parse(doc: RawDocument) -> Document:
    """Build the model for ``doc.source_type``; raises ``pydantic.ValidationError`` if the format is wrong."""

    text = doc.raw_text.strip()
    if doc.source_type == "postmortem":
        return Postmortem.from_markdown(doc.title, doc.metadata, text)
    if doc.source_type == "runbook":
        return Runbook(title=doc.title, sections=split_sections(text))
    if doc.source_type == "doc":
        return TechDoc(title=doc.title, sections=split_sections(text))
    if doc.source_type == "github":
        title, _, body = text.partition("\n")
        kind = "pr" if doc.metadata.get("kind") == "pr" else "commit"
        return CodeChange(kind=kind, ref=doc.source_uri.rstrip("/").rsplit("/", 1)[-1], title=doc.title or title,
                          body=body.strip(), author=doc.author, changed_on=doc.updated_at.date())
    if doc.source_type == "slack":
        return SlackThread(channel=str(doc.metadata.get("channel") or ""), messages=doc.metadata.get("messages") or [],
                           last_message_on=doc.updated_at.date())
    raise ValueError(f"no document model for source_type {doc.source_type!r}")
