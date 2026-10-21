import os
import json
import requests
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

from .base import BaseConnector, RawDocument
from schema import GithubIssues, GithubPR

load_dotenv()
owner = "open-telemetry"
repo = "opentelemetry-demo"

keywords = [
    "paymentservice",
    "payment service",
    "charge.js",
    "paymentfailure"
]
payment_paths = (
    "src/payment/",
    "src/paymentservice/"
)

def fetch_closed_issue(owner: str, repo: str):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "doc-ingest-app",
        "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"
    }
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    results = []
    page = 1
    while True:
        params = {
            "page": page,
            "per_page": 100,
            "state": "closed"
        }
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break
        if not response.json():
            break
        issue = [item for item in response.json() if "pull_request" not in item]
        results.extend(issue)
        page += 1
    return results

def fetch_closed_pr(owner: str, repo: str):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "doc-ingest-app",
        "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"
    }
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    results = []
    page = 1
    while True:
        params = {
            "page": page,
            "per_page": 100,
            "state": "closed"
        }
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break
        if not response.json():
            break
        results.extend(response.json())
        page += 1
    return [pr for pr in results if pr.get("merged_at") is not None]

def create_file(filename: str, function):
    if os.path.exists(filename):
        print(f"Loading from {filename}")
        with open(filename, "r") as file:
            return json.load(file)
    print(f"Fetching from Github")
    data = function(owner, repo)
    with open(filename, "w") as file:
        json.dump(data, file)
    return data

def fetch_changed_file(number: int):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "doc-ingest-app",
        "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"
    }
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}/files"
    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return []
    return response.json()


class GitHubConnector(BaseConnector):
    def __init__(self, sources_path: str | Path | None = None, token: str | None = None) -> None:
        self.sources_path = sources_path
        self.token = token

    async def fetch(self, since: datetime | None = None) -> list[RawDocument]:
        closed_issues = create_file("closed_issues.json", fetch_closed_issue)
        closed_prs = create_file("closed_prs.json", fetch_closed_pr)

        relevant_issues = []
        relevant_prs = []

        for issue in closed_issues:
            text = ((issue.get("title") or "") + " " + (issue.get("body") or "")).lower()
            if any(keyword in text for keyword in keywords):
                relevant_issues.append(issue)

        for pr in closed_prs:
            text = ((pr.get("title") or "") + " " + (pr.get("body") or "")).lower()
            if any(keyword in text for keyword in keywords):
                relevant_prs.append(pr)

        def filter_prs_by_file(owner, repo):
            result = []
            for pr in relevant_prs:
                files = fetch_changed_file(pr["number"])
                if any(file["filename"].startswith(payment_paths) for file in files):
                    result.append(pr)
            return result

        filtered_prs = create_file("filtered_prs.json", filter_prs_by_file)

        issues = [GithubIssues.model_validate(data) for data in relevant_issues]
        prs = [GithubPR.model_validate(data) for data in filtered_prs]

        documents: list[RawDocument] = []
        now = datetime.now(timezone.utc)

        for issue in issues:
            documents.append(RawDocument(
                doc_id=f"github:{owner}/{repo}/issues/{issue.number}",
                source_type="github",
                source_uri=f"https://github.com/{owner}/{repo}/issues/{issue.number}",
                title=issue.title,
                raw_text=f"{issue.title}\n\n{issue.body or ''}".strip(),
                updated_at=now,
                author=owner,
                metadata={"kind": "issue", "number": issue.number}
            ))

        for pr in prs:
            documents.append(RawDocument(
                doc_id=f"github:{owner}/{repo}/pull/{pr.number}",
                source_type="github",
                source_uri=f"https://github.com/{owner}/{repo}/pull/{pr.number}",
                title=pr.title,
                raw_text=f"{pr.title}\n\n{pr.body or ''}".strip(),
                updated_at=now,
                author=owner,
                metadata={"kind": "pr", "number": pr.number, "pr_merged": True}
            ))

        return documents
