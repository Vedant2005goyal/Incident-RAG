from datetime import UTC, datetime

import requests

from .base import BaseConnector, RawDocument, ensure_utc


class SlackConnector(BaseConnector):
    def __init__(self, bot_token: str | None, channel_ids: list[str] | None = None):
        self.bot_token = bot_token
        self.channel_ids = channel_ids or []
        self.headers = {"Authorization": f"Bearer {bot_token}"} if bot_token else {}

    async def fetch(self, since: datetime | None = None) -> list[RawDocument]:
        if not self.bot_token:
            return []

        if self.channel_ids:
            channels = [{"id": channel_id} for channel_id in self.channel_ids]
        else:
            channels = self._get("conversations.list", {"types": "public_channel", "limit": 100}).get("channels", [])

        documents = []
        for channel in channels:
            channel_id = channel["id"]
            channel_name = channel.get("name", channel_id)
            params = {"channel": channel_id, "limit": 100}
            if since:
                params["oldest"] = str(ensure_utc(since).timestamp())

            messages = self._get("conversations.history", params).get("messages", [])
            for message in messages:
                if message.get("thread_ts") != message["ts"]:
                    continue

                thread = self._get(
                    "conversations.replies",
                    {"channel": channel_id, "ts": message["ts"], "limit": 100},
                ).get("messages", [])
                if not thread:
                    continue

                text = "\n".join(f"{item.get('user', 'unknown')}: {item.get('text', '')}" for item in thread)
                documents.append(RawDocument(
                    doc_id=f"slack-{channel_id}-{message['ts']}",
                    source_type="slack",
                    source_uri=f"slack://channel/{channel_id}/thread/{message['ts']}",
                    title=f"Slack Thread in {channel_name}",
                    raw_text=text,
                    updated_at=datetime.fromtimestamp(float(thread[-1]["ts"]), tz=UTC),
                    author=thread[0].get("user", "unknown"),
                    metadata={"channel": channel_name, "messages": thread},
                ))

        return documents

    def _get(self, path: str, params: dict) -> dict:
        response = requests.get(
            f"https://slack.com/api/{path}", headers=self.headers, params=params, timeout=15
        )
        response.raise_for_status()
        data = response.json()
        return data if data.get("ok") else {}

