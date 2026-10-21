"""Tests for the OpenAI-compatible embedder."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.embeddings import BATCH_SIZE, ApiEmbedder
from src.settings import Settings


def fake_client() -> AsyncMock:
    """Answers each request with one vector per input, like Gemini (no "index" field)."""

    async def post(url, headers, timeout, json):
        response = MagicMock()
        response.json.return_value = {"data": [{"embedding": [float(len(text))]} for text in json["input"]]}
        return response

    client = AsyncMock()
    client.post.side_effect = post
    return client


async def test_vectors_come_back_in_input_order_and_in_batches():
    client = fake_client()
    embedder = ApiEmbedder(Settings(_env_file=None, llm_base_url="http://x/v1/", llm_api_key="k"), client)
    texts = ["a" * (n + 1) for n in range(BATCH_SIZE + 2)]
    vectors = await embedder.embed_texts(texts)
    assert vectors == [[float(len(t))] for t in texts]
    assert client.post.call_count == 2
    call = client.post.call_args_list[0]
    assert call.args[0] == "http://x/v1/embeddings"
    assert call.kwargs["headers"] == {"Authorization": "Bearer k"}
    assert call.kwargs["json"]["dimensions"] == 768


async def test_waits_and_retries_when_rate_limited():
    client = fake_client()
    ok = client.post.side_effect
    limited = MagicMock(status_code=429)
    client.post.side_effect = [limited, await ok("u", headers={}, timeout=1, json={"input": ["ab"]})]
    embedder = ApiEmbedder(Settings(_env_file=None, llm_api_key="k"), client)
    with patch("src.embeddings.asyncio.sleep", new=AsyncMock()) as sleep:
        vectors = await embedder.embed_texts(["ab"])
    assert vectors == [[2.0]]
    assert client.post.call_count == 2
    sleep.assert_awaited_once()
