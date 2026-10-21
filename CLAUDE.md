# CLAUDE.md

Question → RAG pipeline → cited answer. One FastAPI service (Python 3.12, `uv`); see `README.md` for the flow.
The owner uses this project for interviews: keep it simple and explainable, and don't add features nobody asked for.

## Commands

```bash
uv run pytest                                   # tests that need live Qdrant or an LLM key skip themselves
uv run pytest tests/unit/test_answer.py         # one file
uv run uvicorn src.main:app --port 8002         # needs Qdrant (docker compose up -d qdrant) + LLM_API_KEY in .env
uv run ask "why did payments fail?"             # CLI, calls POST /ask
uv run --group eval python eval/run_ragas.py    # RAGAS eval against the running service -> eval/results.md (generated)
```

## Layout

- `src/main.py`: endpoints. `retrieve()` is the RAG pipeline (dense + sparse → WRRF → cross-encoder → authority →
  LLMLingua-2 per chunk + ID re-injection). `/ask` = `retrieve()` + `answer.py`.
- `src/answer.py`: the LLM writes the text only; code keeps valid citations; template fallback without an LLM.
- `src/schemas.py`: API request/response models. Logging is plain `logging`, configured in `main.py`.
- `src/embeddings.py`: 768-dim vectors from the API's `/embeddings` (same model for ingest and search).
- Qdrant holds one service only (the service under test, now the OpenTelemetry Demo payment service): don't index
  other services' data. `data/sources.yaml` names the service's repo paths for code / runbooks / docs.
- `src/documents.py`: one Pydantic model per document type (postmortem, runbook, doc, github, slack); `parse()`
  validates the format and `model.chunks()` is that type's chunking. Invalid documents are skipped at ingest.
- Metadata: `Preprocessor.chunk()` puts a `[type · service · date · section]` header on each chunk (embedded);
  `ChunkMetadata.to_payload()` / `from_payload()` in `src/preprocessor.py` are the only writer/reader of the payload.
- `src/connectors/`: one per `POST /ingest {source_type}`: `local_file` (`data/docs/`), `github`
  (`data/sources.yaml`), `slack`.
- Settings in `src/settings.py` / `.env`: one OpenAI-compatible API (`LLM_BASE_URL`, `LLM_API_KEY`; Gemini by default)
  for embeddings and answer, with `EMBEDDING_MODEL`, `ANSWER_LLM_MODEL`.
- Pinned: `qdrant-client<1.10` to match the Qdrant 1.9 server (newer clients removed `.search()`).
