"""Tests for the answer step of /ask."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.answer import NO_DOCS, Answerer
from src.schemas import KnowledgeChunk


def chunk(n: int) -> KnowledgeChunk:
    return KnowledgeChunk(chunk_id=f"doc{n}-chunk-0", source_type="runbook", source_uri=f"confluence://doc{n}",
                          authority_tier="A", recency_score=1.0, content=f"content {n}",
                          retrieved_at=datetime.now(UTC))


def llm_replying(text: str) -> MagicMock:
    llm = MagicMock()
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
    llm.chat.completions.create = AsyncMock(return_value=reply)
    return llm


async def test_no_chunks_does_not_call_the_llm():
    llm = llm_replying("{}")
    result = await Answerer(llm, "m").answer("why?", [])
    assert result.answer == NO_DOCS and result.sources == [] and not result.used_llm
    llm.chat.completions.create.assert_not_called()


async def test_citations_map_to_sources_and_made_up_numbers_are_dropped():
    llm = llm_replying('Here: {"answer": "db-primary is down [2].", "cited": [2, 7]}')
    result = await Answerer(llm, "m").answer("why?", [chunk(1), chunk(2)])
    assert result.answer == "db-primary is down [2]."
    assert [(s.n, s.source_uri) for s in result.sources] == [(2, "confluence://doc2")]
    assert result.used_llm


async def test_llm_failure_falls_back_to_top_document():
    llm = MagicMock()
    llm.chat.completions.create = AsyncMock(side_effect=RuntimeError("down"))
    result = await Answerer(llm, "m").answer("why?", [chunk(1), chunk(2)])
    assert result.answer.startswith("No LLM answer was generated") and "confluence://doc1" in result.answer
    assert [s.n for s in result.sources] == [1] and not result.used_llm


async def test_without_llm_uses_template():
    result = await Answerer().answer("why?", [chunk(1)])
    assert not result.used_llm and result.sources[0].chunk_id == "doc1-chunk-0"
