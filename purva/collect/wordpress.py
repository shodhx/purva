from __future__ import annotations

import time
import urllib.robotparser
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen, Request
from typing import Iterator

import requests
from bs4 import BeautifulSoup

from .base import Collector, Document


class WordpressCollector(Collector):
    name = "wordpress"

    def __init__(
        self,
        source_name: str,
        base_url: str,
        link_selector: str,
        content_selector: str,
        user_agent: str,
        category_paths: list[str] | None = None,
        request_delay: float = 2.0,
        max_pages_per_category: int = 50,
        timeout: float = 20.0,
        respect_robots: bool = True,
    ):
        self.name = source_name
        self.base_url = base_url.rstrip("/")
        self.link_selector = link_selector
        self.content_selector = content_selector
        self.category_paths = category_paths
        self.request_delay = request_delay
        self.max_pages_per_category = max_pages_per_category
        self.timeout = timeout
        self.respect_robots = respect_robots
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _robots_for(self, root: str):
        if root in self._robots:
            return self._robots[root]
        rp = urllib.robotparser.RobotFileParser()
        robots_url = urljoin(root, "/robots.txt")
        try:
            req = Request(robots_url, headers={"User-Agent": self.session.headers["User-Agent"]})
            with urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
            rp.parse(body.splitlines())
        except Exception as e:
            print(f"  [robots] could not read {robots_url} ({e}); proceeding")
            rp = None
        self._robots[root] = rp
        return rp

    def _allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urlparse(url)
        root = f"{parts.scheme}://{parts.netloc}"
        rp = self._robots_for(root)
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

    def _leaves_only(self, cats: list[str]) -> list[str]:
        labels = {self.category_label(c): c for c in cats}
        keep = []
        for label, url in labels.items():
            if label in ("uncategorized",):
                continue
            is_parent = any(other != label and other.startswith(label + "/")
                            for other in labels)
            if not is_parent:
                keep.append(url)
        return keep

    def discover_categories(self) -> list[str]:
        if self.category_paths:
            cats = [urljoin(self.base_url + "/", c.strip("/")) for c in self.category_paths]
            print(f"  [cats] using {len(cats)} configured categories")
            return cats

        found: list[str] = []
        sitemap_urls = [
            urljoin(self.base_url + "/", "category-sitemap.xml"),
            urljoin(self.base_url + "/", "sitemap.xml"),
            urljoin(self.base_url + "/", "sitemap_index.xml"),
        ]
        for sm in sitemap_urls:
            body = self._get(sm)
            if not body:
                continue
            soup = BeautifulSoup(body, "xml")
            locs = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
            cats = [u for u in locs if "/category/" in u]
            if cats:
                found = self._leaves_only(list(dict.fromkeys(cats)))
                print(f"  [cats] discovered {len(found)} leaf categories via {sm}")
                return found
            nested = [u for u in locs if "category" in u and u.endswith(".xml")]
            for n in nested:
                nb = self._get(n)
                if nb:
                    ns = BeautifulSoup(nb, "xml")
                    cats += [loc.get_text(strip=True) for loc in ns.find_all("loc")
                             if "/category/" in loc.get_text()]
            if cats:
                found = list(dict.fromkeys(cats))
                print(f"  [cats] discovered {len(found)} via nested {sm}")
                return found

        home = self._get(self.base_url + "/")
        if home:
            soup = BeautifulSoup(home, "lxml")
            cats = [urljoin(self.base_url, a.get("href"))
                    for a in soup.find_all("a", href=True)
                    if "/category/" in a.get("href")]
            found = list(dict.fromkeys(cats))
            print(f"  [cats] discovered {len(found)} via homepage nav")
        return found

    def _article_links(self, listing_html: str, base_url: str) -> list[str]:
        soup = BeautifulSoup(listing_html, "lxml")
        links = [urljoin(base_url, a.get("href"))
                 for a in soup.select(self.link_selector) if a.get("href")]
        return list(dict.fromkeys(links))

    def _article_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        nodes = soup.select(self.content_selector)
        return "\n".join(n.get_text(separator=" ", strip=True) for n in nodes)

    def category_label(self, cat_url: str) -> str:
        return cat_url.rstrip("/").split("/category/")[-1].strip("/") or cat_url

    def iter_category_articles(self, cat_url: str) -> Iterator[Document]:
        seen: set[str] = set()
        for page in range(1, self.max_pages_per_category + 1):
            page_url = cat_url if page == 1 else urljoin(cat_url.rstrip("/") + "/", f"page/{page}/")
            listing = self._get(page_url)
            if not listing:
                break
            links = [l for l in self._article_links(listing, page_url) if l not in seen]
            if not links:
                break
            for link in links:
                seen.add(link)
                html = self._get(link)
                if not html:
                    continue
                text = self._article_text(html)
                if text.strip():
                    yield Document(url=link, text=text,
                                   meta={"source": self.name,
                                         "category": self.category_label(cat_url)})


    def _sitemap_index_posts(self) -> list[str]:
        idx = self._get(urljoin(self.base_url + "/", "sitemap_index.xml"))
        if not idx:
            return []
        soup = BeautifulSoup(idx, "xml")
        maps = [loc.get_text(strip=True) for loc in soup.find_all("loc")
                if "post-sitemap" in loc.get_text()]
        urls: list[str] = []
        for sm in maps:
            body = self._get(sm)
            if not body:
                continue
            ss = BeautifulSoup(body, "xml")
            urls += [loc.get_text(strip=True) for loc in ss.find_all("loc")
                     if not loc.find_parent("image")]
        return list(dict.fromkeys(urls))

    def iter_sitemap_articles(self, include: list[str], exclude: list[str]) -> Iterator[Document]:
        import re as _re
        inc = [_re.compile(p, _re.IGNORECASE) for p in include] if include else []
        exc = [_re.compile(p, _re.IGNORECASE) for p in exclude] if exclude else []
        urls = self._sitemap_index_posts()
        print(f"  [sitemap] {len(urls)} post urls found")
        kept_urls = []
        for u in urls:
            slug = u.rstrip("/").rsplit("/", 1)[-1]
            if exc and any(p.search(slug) for p in exc):
                continue
            if inc and not any(p.search(slug) for p in inc):
                continue
            kept_urls.append(u)
        print(f"  [sitemap] {len(kept_urls)} urls after slug filters")
        for link in kept_urls:
            html = self._get(link)
            if not html:
                continue
            text = self._article_text(html)
            if text.strip():
                yield Document(url=link, text=text,
                               meta={"source": self.name, "category": "sitemap"})

    def iter_documents(self) -> Iterator[Document]:
        for cat in self.discover_categories():
            yield from self.iter_category_articles(cat)