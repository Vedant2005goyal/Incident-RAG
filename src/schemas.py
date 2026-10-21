"""Request and response bodies of the API."""

from datetime import datetime

from pydantic import BaseModel, Field


class KnowledgeRequest(BaseModel):
    """Input of the retrieval pipeline (POST /knowledge)."""

    query_id: str
    semantic_query: str
    max_chunks: int
    compression_budget_tokens: int


class KnowledgeChunk(BaseModel):
    """One retrieved, compressed chunk and where it came from."""

    chunk_id: str
    source_type: str
    source_uri: str
    authority_tier: str
    recency_score: float
    title: str = ""
    section: str = ""
    content: str
    retrieved_at: datetime


class KnowledgeResult(BaseModel):
    """Output of the retrieval pipeline."""

    query_id: str
    chunks: list[KnowledgeChunk]
    total_tokens_after_compression: int


class AskRequest(BaseModel):
    """A plain-English question, e.g. "why did payments fail?"."""

    question: str = Field(min_length=1)
    max_chunks: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    """A document the answer cites as [n]."""

    n: int
    source_uri: str
    chunk_id: str


class AskResponse(BaseModel):
    """The answer, the sources it cites, and the chunks it was written from."""

    question: str
    answer: str
    sources: list[Source]
    used_llm: bool
    chunks: list[KnowledgeChunk]
