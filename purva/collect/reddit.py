from __future__ import annotations

import time
from typing import Iterator

import requests

from .base import Collector, Document


class RedditCollector(Collector):
    name = "reddit"

    def __init__(
        self,
        source_name: str,
        subreddit: str,
        user_agent: str,
        request_delay: float = 2.0,
        timeout: float = 20.0,
        max_retries: int = 5,
        max_posts: int = 1000,
    ):
        self.name = source_name
        self.subreddit = subreddit
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_posts = max_posts
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def _get(self, url: str, params: dict | None = None):
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    wait = min(10 + 10 * attempt, 60)
                    print(f"  [reddit] 429; waiting {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                time.sleep(self.request_delay)
                return resp.json()
            except requests.RequestException as e:
                wait = min(2 ** attempt, 30)
                print(f"  [reddit] {e}; retry in {wait}s")
                time.sleep(wait)
        print(f"  [reddit] gave up on {url}")
        return None

    def _post_ids(self) -> Iterator[tuple[str, str, str]]:
        after = None
        pulled = 0
        base = f"https://old.reddit.com/r/{self.subreddit}/new.json"
        while pulled < self.max_posts:
            params = {"limit": 100}
            if after:
                params["after"] = after
            data = self._get(base, params)
            if not data:
                break
            children = data.get("data", {}).get("children", [])
            if not children:
                break
            for c in children:
                d = c.get("data", {})
                yield d.get("id", ""), d.get("title", ""), d.get("selftext", "")
                pulled += 1
            after = data.get("data", {}).get("after")
            if not after:
                break

    def _comments(self, post_id: str) -> list[str]:
        url = f"https://old.reddit.com/r/{self.subreddit}/comments/{post_id}.json"
        data = self._get(url)
        if not data or len(data) < 2:
            return []
        out: list[str] = []

        def walk(node):
            kind = node.get("kind")
            d = node.get("data", {})
            if kind == "t1":
                body = d.get("body", "")
                if body and body not in ("[deleted]", "[removed]"):
                    out.append(body)
            replies = d.get("replies")
            if isinstance(replies, dict):
                for child in replies.get("data", {}).get("children", []):
                    walk(child)

        for child in data[1].get("data", {}).get("children", []):
            walk(child)
        return out

    def iter_documents(self) -> Iterator[Document]:
        for pid, title, body in self._post_ids():
            url = f"https://old.reddit.com/r/{self.subreddit}/comments/{pid}"
            if title.strip():
                yield Document(url=url, text=title,
                               meta={"source": self.name, "kind": "title"})
            if body.strip():
                yield Document(url=url, text=body,
                               meta={"source": self.name, "kind": "selftext"})
            for c in self._comments(pid):
                yield Document(url=url, text=c,
                               meta={"source": self.name, "kind": "comment"})