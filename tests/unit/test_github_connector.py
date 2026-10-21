import pytest
from datetime import datetime, UTC
from unittest.mock import MagicMock, patch
from src.connectors.github import GitHubConnector

SOURCES = """
code:
  - repo: acme/shop
    path: src/payment
runbooks:
  - repo: acme/shop
    path: src/payment/runbooks
docs:
  - repo: acme/website
    path: docs/services/payment.md
"""

TREES = {
    "acme/shop": ["src/payment/runbooks/charge-errors.md", "src/payment/runbooks/script.sh", "src/cart/runbooks/x.md"],
    "acme/website": ["docs/services/payment.md", "docs/services/payment.md.bak", "docs/services/cart.md"],
}

FILES = {
    "src/payment/runbooks/charge-errors.md": "---\ntitle: Charge errors\n---\nCheck the card processor.",
    "docs/services/payment.md": "The payment service charges the card.",
}


def _response(payload=None, text=""):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    response.text = text
    return response


def _commit(sha, message, parents=1, author_type="User", date="2026-10-02T10:00:00Z"):
    return {"sha": sha * 40, "html_url": f"https://github.com/acme/shop/commit/{sha}", "parents": [{}] * parents,
            "author": {"type": author_type},
            "commit": {"message": message, "author": {"name": "dev", "date": date}, "committer": {"date": date}}}


def _github(url, params=None):
    if url.startswith("https://raw.githubusercontent.com/"):
        return _response(text=FILES[url.split("/main/", 1)[1]])
    if url.endswith("/commits"):
        assert params["path"] == "src/payment"
        return _response([
            _commit("a", "Retry card charges (#12)\n\nCharges are retried 3 times."),
            _commit("b", "Fix typo in payment logs"),
            _commit("c", "Merge branch main", parents=2),
            _commit("d", "chore(deps): bump grpc (#13)", author_type="Bot"),
        ])
    if "/git/trees/" in url:
        repo = url.removeprefix("/repos/").split("/git/")[0]
        return _response({"tree": [{"path": path, "type": "blob"} for path in TREES[repo]]})
    return _response({"default_branch": "main", "pushed_at": "2026-10-03T10:00:00Z"})  # /repos/{repo}


@pytest.fixture
def connector(tmp_path):
    sources = tmp_path / "sources.yaml"
    sources.write_text(SOURCES)
    return GitHubConnector(sources)


@pytest.mark.asyncio
async def test_github_fetch_no_sources(tmp_path):
    assert await GitHubConnector(tmp_path / "missing.yaml").fetch(None) == []


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_github_fetch_only_the_service(mock_get, connector):
    mock_get.side_effect = _github

    docs = {doc.doc_id: doc for doc in await connector.fetch(since=None)}

    # A path is a folder or a single file; only .md files; the section sets the type
    runbook = docs["github:acme/shop/src/payment/runbooks/charge-errors.md"]
    assert runbook.source_type == "runbook"
    assert runbook.title == "Charge errors"
    doc = docs["github:acme/website/docs/services/payment.md"]
    assert doc.source_type == "doc"
    assert doc.source_uri == "https://github.com/acme/website/blob/main/docs/services/payment.md"

    # A squash-merged commit is stored as its PR; merge commits and bot commits are skipped
    pr = docs["github:acme/shop/pull/12"]
    assert pr.source_type == "github"
    assert pr.source_uri == "https://github.com/acme/shop/pull/12"
    assert pr.metadata == {"kind": "pr", "pr_merged": True}
    assert "retried 3 times" in pr.raw_text
    assert docs["github:acme/shop/commit/bbbbbbbbbbbb"].metadata == {"kind": "commit"}

    assert len(docs) == 4


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_github_fetch_since_skips_unchanged(mock_get, connector):
    mock_get.side_effect = _github

    docs = await connector.fetch(since=datetime(2026, 10, 5, tzinfo=UTC))

    assert docs == []  # nothing pushed or committed after Oct 5
