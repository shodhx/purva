from __future__ import annotations

import time
import urllib.robotparser
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen, Request
from typing import Iterator

import requests
from bs4 import BeautifulSoup

from .base import Collector, Document


class NingCollector(Collector):
    name = "ning"

    def __init__(
        self,
        source_name: str,
        group_url: str,
        user_agent: str,
        link_selector: str = ".topic h3 a",
        content_selector: str = ".xg_user_generated p",
        request_delay: float = 2.0,
        max_pages: int = 100,
        timeout: float = 20.0,
        respect_robots: bool = True,
    ):
        self.name = source_name
        self.group_url = group_url.rstrip("/")
        self.link_selector = link_selector
        self.content_selector = content_selector
        self.request_delay = request_delay
        self.max_pages = max_pages
        self.timeout = timeout
        self.respect_robots = respect_robots
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._rp = None
        self._rp_loaded = False

    def _robots(self):
        if self._rp_loaded:
            return self._rp
        self._rp_loaded = True
        parts = urlparse(self.group_url)
        root = f"{parts.scheme}://{parts.netloc}"
        rp = urllib.robotparser.RobotFileParser()
        try:
            req = Request(urljoin(root, "/robots.txt"),
                          headers={"User-Agent": self.session.headers["User-Agent"]})
            with urlopen(req, timeout=self.timeout) as resp:
                rp.parse(resp.read().decode("utf-8", errors="ignore").splitlines())
            self._rp = rp
        except Exception as e:
            print(f"  [robots] unreadable ({e}); proceeding")
            self._rp = None
        return self._rp

    def _allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        rp = self._robots()
        if rp is None:
            return True
        return rp.can_fetch(self.session.headers["User-Agent"], url)

    def _get(self, url: str) -> str | None:
        if not self._allowed(url):
            print(f"  [robots] disallowed: {url}")
            return None
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [http] {e} :: {url}")
            return None
        finally:
            time.sleep(self.request_delay)
        resp.encoding = resp.apparent_encoding or resp.encoding
        return resp.text

    def _topic_links(self, html: str) -> list[str]:
        soup = BeautifulSoup(html, "lxml")
        links = []
        for a in soup.select(self.link_selector):
            href = a.get("href")
            if href and "/forum/topics/" in href:
                links.append(urljoin(self.group_url, href))
        return list(dict.fromkeys(links))

    def _post_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        nodes = soup.select(self.content_selector)
        return "\n".join(n.get_text(separator=" ", strip=True) for n in nodes)

    def _listing_pages(self) -> Iterator[str]:
        forum_url = self.group_url + "/forum"
        for page in range(1, self.max_pages + 1):
            url = forum_url if page == 1 else f"{forum_url}?page={page}"
            html = self._get(url)
            if not html:
                break
            links = self._topic_links(html)
            if not links:
                break
            print(f"  [page {page}] {len(links)} topics")
            yield from links

    def iter_documents(self) -> Iterator[Document]:
        seen: set[str] = set()
        for link in self._listing_pages():
            if link in seen:
                continue
            seen.add(link)
            html = self._get(link)
            if not html:
                continue
            text = self._post_text(html)
            if text.strip():
                yield Document(url=link, text=text,
                               meta={"source": self.name, "category": "forum_post"})