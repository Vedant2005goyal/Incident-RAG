from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Read from environment variables or `.env`: each field's variable is its name in capitals (e.g. LLM_API_KEY)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")  # .env may hold keys we don't use

    log_level: str = "INFO"

    # One OpenAI-compatible API (Gemini by default) for the embeddings and the answer
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    llm_api_key: str = ""
    embedding_model: str = "gemini-embedding-001"
    answer_llm_model: str = "gemini-3.5-flash"
    eval_llm_model: str = "gemini-3.5-flash-lite"  # RAGAS judge (eval/); another model, so it has its own rate limit

    qdrant_url: str = "http://localhost:6333"
    dense_collection: str = "nexgen_dense"
    sparse_collection: str = "nexgen_bm25_terms"

    # Retrieval (these models run locally)
    wrrf_k: int = 60
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llmlingua2_model: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    default_compression_budget_tokens: int = 2000

    # What gets ingested
    service_name: str = "payment"  # the service under test
    docs_path: str = "data/docs"
    github_sources_path: str = "data/sources.yaml"
    github_token: str | None = None
    slack_bot_token: str | None = None
    slack_channels: str = ""  # comma-separated channel IDs
