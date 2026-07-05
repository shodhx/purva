from __future__ import annotations

import time
from typing import Iterator

import requests

from .base import Collector, Document


class WikiCollector(Collector):
    name = "wikipedia"

    def __init__(
        self,
        source_name: str,
        api_url: str,
        user_agent: str,
        request_delay: float = 0.5,
        batch: int = 20,
        mode: str = "random",
        timeout: float = 20.0,
        max_retries: int = 4,
    ):
        self.name = source_name
        self.api_url = api_url
        self.request_delay = request_delay
        self.batch = batch
        self.mode = mode
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def _get(self, params: dict) -> dict | None:
        params = {**params, "format": "json"}
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(self.api_url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                time.sleep(self.request_delay)
                return resp.json()
            except requests.RequestException as e:
                wait = min(2 ** attempt, 20)
                print(f"  [wiki] {e}; retry in {wait}s")
                time.sleep(wait)
        print("  [wiki] giving up on a request after retries")
        return None

    def _random_titles(self, n: int) -> list[str]:
        data = self._get({
            "action": "query",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": min(n, 20),
        })
        if not data:
            return []
        return [p["title"] for p in data.get("query", {}).get("random", [])]

    def _extracts(self, titles: list[str]) -> dict[str, str]:
        if not titles:
            return {}
        data = self._get({
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "titles": "|".join(titles),
        })
        out: dict[str, str] = {}
        if not data:
            return out
        for page in data.get("query", {}).get("pages", {}).values():
            text = page.get("extract", "")
            if text and text.strip():
                out[page.get("title", "")] = text
        return out

    def _all_titles(self) -> Iterator[str]:
        cont = None
        while True:
            params = {
                "action": "query",
                "list": "allpages",
                "apnamespace": 0,
                "aplimit": 500,
            }
            if cont:
                params["apcontinue"] = cont
            data = self._get(params)
            if not data:
                break
            pages = data.get("query", {}).get("allpages", [])
            for p in pages:
                yield p.get("title", "")
            cont = data.get("continue", {}).get("apcontinue")
            if not cont:
                break

    def iter_all_documents(self) -> Iterator[Document]:
        batch: list[str] = []
        for title in self._all_titles():
            if not title:
                continue
            batch.append(title)
            if len(batch) >= self.batch:
                for t, text in self._extracts(batch).items():
                    yield Document(url=f"{self.api_url}?title={t}", text=text,
                                   meta={"source": self.name, "category": "wikipedia"})
                batch = []
        if batch:
            for t, text in self._extracts(batch).items():
                yield Document(url=f"{self.api_url}?title={t}", text=text,
                               meta={"source": self.name, "category": "wikipedia"})

    def iter_documents(self) -> Iterator[Document]:
        if self.mode == "all":
            yield from self.iter_all_documents()
            return
        seen: set[str] = set()
        empty_streak = 0
        while empty_streak < 10:
            titles = [t for t in self._random_titles(self.batch) if t not in seen]
            for t in titles:
                seen.add(t)
            extracts = self._extracts(titles)
            if not extracts:
                empty_streak += 1
                continue
            empty_streak = 0
            for title, text in extracts.items():
                yield Document(url=f"{self.api_url}?title={title}", text=text,
                               meta={"source": self.name, "category": "wikipedia"})