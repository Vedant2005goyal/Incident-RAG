import pytest
from unittest.mock import MagicMock, patch
from src.connectors.slack import SlackConnector

@pytest.fixture
def slack_connector():
    return SlackConnector(bot_token="xoxb-test-token")

@pytest.mark.asyncio
async def test_slack_fetch_no_auth():
    connector = SlackConnector(bot_token=None)
    docs = await connector.fetch(None)
    assert len(docs) == 0

@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_slack_fetch_threads(mock_get, slack_connector):
    def mock_get_side_effect(url, params=None, **kwargs):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        
        if "conversations.list" in url:
            mock_resp.json.return_value = {
                "ok": True,
                "channels": [{"id": "C123", "name": "general"}]
            }
        elif "conversations.history" in url:
            mock_resp.json.return_value = {
                "ok": True,
                "messages": [
                    {"ts": "100.0", "thread_ts": "100.0", "user": "U1", "text": "Parent message"}
                ]
            }
        elif "conversations.replies" in url:
            mock_resp.json.return_value = {
                "ok": True,
                "messages": [
                    {"ts": "100.0", "user": "U1", "text": "Parent message"},
                    {"ts": "101.0", "user": "U2", "text": "Reply message"}
                ]
            }
            
        return mock_resp

    mock_get.side_effect = mock_get_side_effect

    docs = await slack_connector.fetch(since=None)
    
    assert len(docs) == 1
    doc = docs[0]
    
    assert doc.doc_id == "slack-C123-100.0"
    assert doc.raw_text == "U1: Parent message\nU2: Reply message"  # one line per message
    assert doc.author == "U1"
    assert doc.metadata["channel"] == "general"
    assert doc.metadata["messages"] == [{"user": "U1", "text": "Parent message"}, {"user": "U2", "text": "Reply message"}]



@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_slack_fetch_only_configured_channels(mock_get):
    def mock_get_side_effect(url, params=None, **kwargs):
        assert "conversations.list" not in url  # channels come from SLACK_CHANNELS
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"ok": True, "messages": []}
        return mock_resp

    mock_get.side_effect = mock_get_side_effect

    docs = await SlackConnector(bot_token="xoxb-test-token", channel_ids=["C999"]).fetch(since=None)

    assert docs == []
    assert mock_get.call_args.kwargs["params"]["channel"] == "C999"
