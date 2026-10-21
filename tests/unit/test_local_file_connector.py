from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from src.connectors.local_file import LocalFileConnector


def test_local_file_connector_reads_markdown_with_front_matter_only(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "runbook.md").write_text(
        "---\n"
        "title: Payments DB Failover\n"
        "author: ops-team\n"
        "source_uri: confluence://runbooks/payments-db-failover\n"
        "updated_at: 2026-04-03T11:30:00Z\n"
        "doc_id: runbook-001\n"
        "---\n"
        "Trigger failover to db-replica-1.\n",
        encoding="utf-8",
    )
    (docs_dir / "notes.txt").write_text("Not Markdown, so not read.\n", encoding="utf-8")

    documents = asyncio.run(LocalFileConnector(docs_dir).fetch(since=None))

    assert len(documents) == 1
    markdown_doc = documents[0]
    assert markdown_doc.doc_id == "runbook-001"
    assert markdown_doc.source_type == "doc"
    assert markdown_doc.title == "Payments DB Failover"
    assert markdown_doc.author == "ops-team"
    assert markdown_doc.source_uri == "confluence://runbooks/payments-db-failover"
    assert markdown_doc.raw_text == "Trigger failover to db-replica-1."
    assert markdown_doc.updated_at == datetime(2026, 4, 3, 11, 30, tzinfo=UTC)


def test_local_file_connector_fetch_respects_since_filter(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "old.md").write_text(
        "---\nupdated_at: 2026-04-01T00:00:00Z\n---\nOld content.\n",
        encoding="utf-8",
    )
    (docs_dir / "new.md").write_text(
        "---\nupdated_at: 2026-04-10T00:00:00Z\n---\nNew content.\n",
        encoding="utf-8",
    )

    connector = LocalFileConnector(docs_dir)

    documents = asyncio.run(connector.fetch(since=datetime(2026, 4, 5, tzinfo=UTC)))

    assert [document.doc_id for document in documents] == ["new"]


def test_local_file_connector_type_comes_from_folder_then_front_matter(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    for folder in ("postmortems", "runbooks", "technical"):
        (docs_dir / folder).mkdir(parents=True)
    (docs_dir / "postmortems" / "outage.md").write_text("The bank API timed out.\n", encoding="utf-8")
    (docs_dir / "runbooks" / "db-down.md").write_text("Check the primary.\n", encoding="utf-8")
    (docs_dir / "technical" / "payment.md").write_text("Payment calls the bank.\n", encoding="utf-8")
    (docs_dir / "runbooks" / "really-a-postmortem.md").write_text(
        "---\nsource_type: postmortem\n---\nThe disk filled up.\n", encoding="utf-8"
    )

    documents = asyncio.run(LocalFileConnector(docs_dir).fetch(since=None))

    assert {document.doc_id: document.source_type for document in documents} == {
        "outage": "postmortem",
        "db-down": "runbook",
        "payment": "doc",
        "really-a-postmortem": "postmortem",
    }
