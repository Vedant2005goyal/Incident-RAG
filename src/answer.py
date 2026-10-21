"""Last step of /ask: write a short answer from the retrieved chunks and cite them."""

import json
import logging
import re
from pathlib import Path

from openai import AsyncOpenAI

from src.schemas import AskResponse, KnowledgeChunk, Source

logger = logging.getLogger(__name__)

PROMPT = (Path(__file__).resolve().parent.parent / "prompts" / "answer.txt").read_text()
NO_DOCS = "No runbook or document covers this question."


class Answerer:
    """
    The LLM only writes the text. Which documents exist and which ones may be cited
    is decided in code: a citation number that matches no snippet is dropped.
    Without an LLM (no key, or the call fails) the top snippet is returned as-is.
    """

    def __init__(self, llm: AsyncOpenAI | None = None, model: str = "") -> None:
        self.llm = llm
        self.model = model

    async def answer(self, question: str, chunks: list[KnowledgeChunk]) -> AskResponse:
        """Answer ``question`` using only ``chunks`` (snippet [n] is chunks[n-1])."""
        if not chunks:
            return self._response(question, NO_DOCS, [], chunks, used_llm=False)
        if self.llm is not None:
            try:
                text, cited = await self._ask_llm(question, chunks)
                return self._response(question, text, cited, chunks, used_llm=True)
            except Exception as exc:  # LLM down or bad reply: fall back to the template
                logger.warning("answer LLM failed, using template: %s", exc)
        top = chunks[0]
        text = f"No LLM answer was generated. The most relevant document is [1] {top.source_uri}: {top.content}"
        return self._response(question, text, [1], chunks, used_llm=False)

    async def _ask_llm(self, question: str, chunks: list[KnowledgeChunk]) -> tuple[str, list[int]]:
        snippets = "\n\n".join(f"[{n}] ({c.source_uri})\n{c.content}" for n, c in enumerate(chunks, start=1))
        response = await self.llm.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nSnippets:\n{snippets}"},
            ],
        )
        content = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", content, re.DOTALL)  # tolerate text around the JSON
        if match is None:
            raise ValueError(f"no JSON in LLM reply: {content[:200]!r}")
        data = json.loads(match.group(0))
        cited = [int(n) for n in data.get("cited", []) if str(n).isdigit()]
        return str(data["answer"]), cited

    @staticmethod
    def _response(
        question: str, text: str, cited: list[int], chunks: list[KnowledgeChunk], used_llm: bool
    ) -> AskResponse:
        valid = sorted({n for n in cited if 1 <= n <= len(chunks)})
        sources = [Source(n=n, source_uri=chunks[n - 1].source_uri, chunk_id=chunks[n - 1].chunk_id) for n in valid]
        return AskResponse(question=question, answer=text, sources=sources, used_llm=used_llm, chunks=chunks)
