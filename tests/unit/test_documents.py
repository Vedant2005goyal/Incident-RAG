from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.connectors.base import RawDocument
from src.documents import CHUNK_WORDS, parse, split_sections, windows

POSTMORTEM = """## Summary
Every payment charge failed.

## Impact
No orders could be placed.

## Timeline
12:40 errors start. 12:55 flag turned off.

## Root cause
The paymentFailure feature flag was set to 100%.

## Trigger
The flag was changed in the flagd UI.

## Detection
Payment logs: Payment request failed. Invalid token.

## Resolution
Set paymentFailure back to off.

## Lessons and action items
Alert on payment error rate.

## Sources
- internal
"""


def raw(source_type, text, title="Title", metadata=None, source_uri="uri"):
    return RawDocument(doc_id="d", source_type=source_type, source_uri=source_uri, title=title, raw_text=text,
                       updated_at=datetime(2026, 10, 17, tzinfo=UTC),
                       author="dev", metadata=metadata or {})


def test_split_sections_keeps_text_before_the_first_heading():
    sections = split_sections("intro\n# A\nbody a\n## B\n\n## C\nbody c")
    assert [(s.heading, s.body) for s in sections] == [("", "intro"), ("A", "body a"), ("C", "body c")]


def test_windows_cover_the_whole_text():
    words = [f"w{i}" for i in range(1000)]
    parts = windows(" ".join(words))
    assert len(parts) == 3
    assert parts[0].split()[0] == "w0" and parts[-1].split()[-1] == "w999"
    assert windows("short text") == ["short text"]


def test_postmortem_is_chunked_by_question():
    doc = parse(raw("postmortem", POSTMORTEM, title="Payment charges failing",
                    metadata={"service": "payment", "incident_date": "2026-10-17", "severity": "major"}))

    assert doc.service == "payment" and doc.root_cause.startswith("The paymentFailure")
    chunks = [c.text for c in doc.chunks()]
    assert [c.section for c in doc.chunks()] == ["What happened", "Timeline", "Why it happened", "How it was fixed"]
    assert doc.metadata() == {"incident_date": "2026-10-17", "severity": "major"}
    assert [c.split("\n")[0] for c in chunks] == [
        "Payment charges failing — What happened",
        "Payment charges failing — Timeline",
        "Payment charges failing — Why it happened",
        "Payment charges failing — How it was fixed",
    ]
    assert "Root cause:" in chunks[2] and "Invalid token" in chunks[2]  # detection travels with the cause
    assert "internal" not in "".join(chunks)  # ## Sources is not indexed


def test_postmortem_without_root_cause_is_rejected():
    with pytest.raises(ValidationError):
        parse(raw("postmortem", "## Summary\nIt broke.\n## Resolution\nFixed it."))


def test_runbook_gets_one_chunk_per_section():
    chunks = parse(raw("runbook", "## Meaning\nPods restart.\n## Mitigation\nRoll back.", title="PodCrashLooping")).chunks()
    assert [(c.section, c.text) for c in chunks] == [("Meaning", "PodCrashLooping — Meaning\n\nPods restart."),
                                                     ("Mitigation", "PodCrashLooping — Mitigation\n\nRoll back.")]


def test_doc_packs_short_sections_together_and_splits_at_the_limit():
    short = parse(raw("doc", "## A\none\n## B\ntwo", title="Payment")).chunks()
    assert [(c.section, c.text) for c in short] == [("A, B", "Payment\n\n## A\none\n\n## B\ntwo")]

    long_section = " ".join(["word"] * (CHUNK_WORDS - 10))
    chunks = parse(raw("doc", f"## A\n{long_section}\n## B\n{long_section}", title="Payment")).chunks()
    assert len(chunks) == 2 and chunks[1].text.startswith("Payment\n\n## B") and chunks[1].section == "B"


def test_code_change_is_one_chunk_named_after_the_pr():
    doc = parse(raw("github", "Retry card charges (#12)\n\nCharges are retried 3 times.", title="Retry card charges (#12)",
                    metadata={"kind": "pr"}, source_uri="https://github.com/acme/shop/pull/12"))
    assert doc.kind == "pr" and doc.ref == "12"
    assert [c.text for c in doc.chunks()] == ["PR #12: Retry card charges (#12)\n\nCharges are retried 3 times."]
    assert doc.metadata() == {"kind": "pr", "ref": "12", "author": "dev", "changed_on": "2026-10-17"}


def test_slack_thread_is_chunked_in_overlapping_message_windows():
    messages = [{"user": f"U{i}", "text": f"message {i}"} for i in range(12)]
    thread = parse(raw("slack", "", metadata={"channel": "payments", "messages": messages}))
    assert thread.metadata() == {"channel": "payments", "message_count": 12, "last_message_on": "2026-10-17"}
    chunks = [c.text for c in thread.chunks()]
    assert len(chunks) == 2
    assert chunks[0].startswith("Slack thread in #payments\n\nU0: message 0")
    assert "U8: message 8" in chunks[0] and chunks[1].split("\n\n")[1].startswith("U8: message 8")  # 2-message overlap


def test_slack_thread_without_messages_is_rejected():
    with pytest.raises(ValidationError):
        parse(raw("slack", "", metadata={"channel": "payments"}))
