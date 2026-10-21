from src.settings import Settings


def test_settings_defaults():
    settings = Settings(_env_file=None)  # defaults, not the local .env

    assert settings.log_level == "INFO"
    assert settings.llm_base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert settings.embedding_model == "gemini-embedding-001"
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.dense_collection == "nexgen_dense"
    assert settings.sparse_collection == "nexgen_bm25_terms"
    assert settings.docs_path == "data/docs"
