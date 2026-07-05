from __future__ import annotations

import time
from typing import Iterator

import requests
from bs4 import BeautifulSoup

from .base import Collector, Document


class BloggerCollector(Collector):
    name = "blogger"

    def __init__(
        self,
        source_name: str,
        base_url: str,
        user_agent: str,
        request_delay: float = 1.0,
        batch: int = 150,
        timeout: float = 20.0,
        max_retries: int = 4,
    ):
        self.name = source_name
        self.base_url = base_url.rstrip("/")
        self.request_delay = request_delay
        self.batch = batch
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def _get_feed(self, start_index: int) -> dict | None:
        url = f"{self.base_url}/feeds/posts/default"
        params = {"alt": "json", "max-results": self.batch, "start-index": start_index}
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                time.sleep(self.request_delay)
                return resp.json()
            except requests.RequestException as e:
                wait = min(2 ** attempt, 20)
                print(f"  [blogger] {e}; retry in {wait}s")
                time.sleep(wait)
        return None

    def _html_to_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text(separator="\n", strip=True)

    def iter_documents(self) -> Iterator[Document]:
        start = 1
        total = None
        while True:
            data = self._get_feed(start)
            if not data:
                break
            feed = data.get("feed", {})
            if total is None:
                total = int(feed.get("openSearch$totalResults", {}).get("$t", "0"))
                print(f"  [blogger] total posts reported: {total}")
            entries = feed.get("entry", [])
            if not entries:
                break
            print(f"  [blogger] batch at start-index {start}: {len(entries)} posts")
            for e in entries:
                title = e.get("title", {}).get("$t", "")
                content_html = e.get("content", {}).get("$t", "") or e.get("summary", {}).get("$t", "")
                url = ""
                for l in e.get("link", []):
                    if l.get("rel") == "alternate":
                        url = l.get("href", "")
                        break
                cats = [c.get("term", "") for c in e.get("category", [])]
                text = (title + "\n" + self._html_to_text(content_html)).strip()
                if text:
                    yield Document(url=url or self.base_url, text=text,
                                   meta={"source": self.name,
                                         "category": cats[0] if cats else None})
            start += len(entries)
            if total and start > total:
                break