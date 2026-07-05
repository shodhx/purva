from __future__ import annotations

import time
from typing import Iterator

import requests

from .base import Collector, Document


class KavitaKoshCollector(Collector):
    name = "kavitakosh"

    def __init__(
        self,
        source_name: str,
        api_url: str,
        root_category: str,
        user_agent: str,
        request_delay: float = 1.0,
        timeout: float = 20.0,
        max_retries: int = 4,
        max_depth: int = 3,
    ):
        self.name = source_name
        self.api_url = api_url
        self.root_category = root_category
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_depth = max_depth
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
                print(f"  [kk] {e}; retry in {wait}s")
                time.sleep(wait)
        return None

    def _category_members(self, category: str) -> tuple[list[str], list[str]]:
        pages: list[str] = []
        subcats: list[str] = []
        cont = None
        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category,
                "cmlimit": 500,
            }
            if cont:
                params["cmcontinue"] = cont
            data = self._get(params)
            if not data:
                break
            for m in data.get("query", {}).get("categorymembers", []):
                title = m.get("title", "")
                if m.get("ns") == 14:
                    subcats.append(title)
                elif m.get("ns") == 0:
                    pages.append(title)
            cont = data.get("continue", {}).get("cmcontinue")
            if not cont:
                break
        return pages, subcats

    def _extract(self, titles: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for i in range(0, len(titles), 10):
            batch = titles[i:i + 10]
            data = self._get({
                "action": "query",
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "titles": "|".join(batch),
            })
            if not data:
                continue
            for page in data.get("query", {}).get("pages", {}).values():
                revs = page.get("revisions") or []
                if not revs:
                    continue
                rev = revs[0]
                text = ""
                slots = rev.get("slots")
                if slots:
                    text = slots.get("main", {}).get("*", "") or slots.get("main", {}).get("content", "")
                if not text:
                    text = rev.get("*", "") or rev.get("content", "")
                if text and text.strip():
                    out[page.get("title", "")] = text
        return out

    def iter_documents(self) -> Iterator[Document]:
        seen_cats: set[str] = set()
        seen_pages: set[str] = set()
        frontier = [(self.root_category, 0)]
        while frontier:
            cat, depth = frontier.pop(0)
            if cat in seen_cats or depth > self.max_depth:
                continue
            seen_cats.add(cat)
            pages, subcats = self._category_members(cat)
            print(f"  [kk] {cat}: {len(pages)} pages, {len(subcats)} subcategories (depth {depth})")
            for sc in subcats:
                if sc not in seen_cats:
                    frontier.append((sc, depth + 1))
            new_pages = [p for p in pages if p not in seen_pages]
            for p in new_pages:
                seen_pages.add(p)
            extracts = self._extract(new_pages)
            for title, text in extracts.items():
                yield Document(
                    url=f"{self.api_url.replace('/api.php','/index.php')}?title={title}",
                    text=text,
                    meta={"source": self.name, "category": cat.replace("Category:", "").replace("श्रेणी:", "")},
                )