"""The API. `lifespan` builds every component once at startup; `retrieve` is the RAG pipeline."""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import httpx
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, QdrantClient

from .answer import Answerer
from .authority import AuthorityScorer, recency
from .compactor import LLMLingua2Compactor
from .connectors.github import GitHubConnector
from .connectors.local_file import LocalFileConnector
from .connectors.slack import SlackConnector
from .dense import DenseRetriever
from .embeddings import ApiEmbedder
from .fusion import WRRFFusion
from .id_preservation import TechnicalIDPreservationLayer
from .ingest_service import IngestRequest, IngestResponse, IngestService
from .preprocessor import Preprocessor
from .reranker import CrossEncoderReranker
from .schemas import AskRequest, AskResponse, KnowledgeChunk, KnowledgeRequest, KnowledgeResult
from .settings import Settings
from .sparse import SparseRetriever

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    app.state.settings = settings

    # Ingest: one connector per source
    slack_channels = [c.strip() for c in settings.slack_channels.split(",") if c.strip()]
    app.state.ingest_service = IngestService(
        qdrant_client=QdrantClient(url=settings.qdrant_url),
        connectors={
            "local_file": LocalFileConnector(settings.docs_path),  # data/docs/
            "github": GitHubConnector(settings.github_sources_path, settings.github_token),  # data/sources.yaml
            "slack": SlackConnector(settings.slack_bot_token, slack_channels),
        },
        preprocessor=Preprocessor(service=settings.service_name),
        embedder=ApiEmbedder(settings),
        dense_collection=settings.dense_collection,
        sparse_collection=settings.sparse_collection,
    )

    # Retrieval (the models load here, once)
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    http = httpx.AsyncClient()
    app.state.dense_retriever = DenseRetriever(qdrant, http, settings)
    app.state.sparse_retriever = SparseRetriever(qdrant, settings)
    app.state.fusion = WRRFFusion(settings)
    app.state.reranker = CrossEncoderReranker(settings)
    app.state.authority_scorer = AuthorityScorer()
    app.state.compactor = LLMLingua2Compactor(model_name=settings.llmlingua2_model)
    app.state.id_preservation = TechnicalIDPreservationLayer()

    # Answer: without an API key there is no LLM, and a template answer is returned
    llm = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url, max_retries=3) if settings.llm_api_key else None
    app.state.answerer = Answerer(llm, settings.answer_llm_model)

    yield
    await http.aclose()
    await qdrant.close()


app = FastAPI(title="incident-rag", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(request: IngestRequest) -> IngestResponse:
    """Fetch one source's documents, chunk and embed them, and store them in Qdrant."""
    if request.source_type not in app.state.ingest_service.connectors:
        raise HTTPException(status_code=400, detail=f"Unsupported source_type: {request.source_type}")
    return await app.state.ingest_service.ingest(request)


@app.post("/ask")
async def ask(request: AskRequest) -> AskResponse:
    """Answer a question like "why did payments fail?" from the retrieved documents, with citations."""
    knowledge_request = KnowledgeRequest(
        query_id=str(uuid.uuid4()),
        semantic_query=request.question,
        max_chunks=request.max_chunks,
        compression_budget_tokens=app.state.settings.default_compression_budget_tokens,
    )
    result = await retrieve(knowledge_request)
    return await app.state.answerer.answer(request.question, result.chunks)


@app.post("/knowledge")
async def knowledge(request: KnowledgeRequest) -> KnowledgeResult:
    """The retrieved, compressed chunks without an answer (useful to see what search found)."""
    return await retrieve(request)


async def retrieve(request: KnowledgeRequest) -> KnowledgeResult:
    """The RAG pipeline: search two ways, fuse, re-rank, score, keep the top chunks, compress them."""
    state = app.state

    # 1. Search by meaning (dense) and by exact words (sparse), at the same time
    dense, sparse = await asyncio.gather(state.dense_retriever.retrieve(request), state.sparse_retriever.retrieve(request))

    # 2. Merge the two result lists
    w_dense, w_sparse = state.fusion.classify_query(request.semantic_query)
    fused = state.fusion.fuse(dense, sparse, w_dense, w_sparse)

    # 3. Re-rank the best candidates with the cross-encoder, 4. boost trusted and recent ones, 5. keep the top
    reranked = state.reranker.rerank(request.semantic_query, fused[: request.max_chunks * 2])
    top = state.authority_scorer.score(reranked)[: request.max_chunks]

    # 6. Compress each chunk to its share of the token budget, and put back any technical ID it lost
    budget = max(1, request.compression_budget_tokens // max(1, len(top)))
    now = datetime.now(UTC)
    chunks = []
    for chunk in top:
        compressed = state.compactor.compress([chunk.content], budget_tokens=budget)
        result = state.id_preservation.verify_and_reinject([chunk.content], compressed)
        if result.total_reinjections:
            logger.info("put back %d technical IDs: %s", result.total_reinjections, result.reinjected_tags)
        if result.text.strip():
            chunks.append(KnowledgeChunk(
                chunk_id=chunk.chunk_id,
                source_type=chunk.metadata.source_type,
                source_uri=chunk.metadata.source_uri,
                authority_tier=chunk.metadata.authority_tier,
                recency_score=recency(chunk.metadata.created_at),
                title=chunk.metadata.title,
                section=chunk.metadata.section,
                content=result.text,
                retrieved_at=now,
            ))

    return KnowledgeResult(
        query_id=request.query_id,
        chunks=chunks,
        total_tokens_after_compression=sum(len(c.content.split()) for c in chunks),
    )
