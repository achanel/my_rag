"""Data collection and preparation (chapter 2, part 1).

Reimplementation of ``Chapter02/1_Data_collection_preparation.ipynb`` from the
reference book: fetch the space-exploration Wikipedia articles, strip the
navigation/reference scaffolding, and write every article to a single Markdown
corpus file (the book writes plain ``llm.txt``; a heading per article keeps the
sources distinguishable). The next chapter 2 stage (embeddings) reads that file.

The book fetches with ``requests``, parses with ``beautifulsoup4``, keeps only
the ``mw-parser-output`` container, and drops everything from the "References",
"Bibliography", "External links", and "See also" headings onward. Both
dependencies are already on the core stack.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import requests
from bs4 import BeautifulSoup, Tag

#: Wikipedia articles collected by the book's chapter 2 notebook.
WIKI_URLS: list[str] = [
    "https://en.wikipedia.org/wiki/Space_exploration",
    "https://en.wikipedia.org/wiki/Apollo_program",
    "https://en.wikipedia.org/wiki/Hubble_Space_Telescope",
    "https://en.wikipedia.org/wiki/Mars_rover",
    "https://en.wikipedia.org/wiki/International_Space_Station",
    "https://en.wikipedia.org/wiki/SpaceX",
    "https://en.wikipedia.org/wiki/Juno_(spacecraft)",
    "https://en.wikipedia.org/wiki/Voyager_program",
    "https://en.wikipedia.org/wiki/Galileo_(spacecraft)",
    "https://en.wikipedia.org/wiki/Kepler_Space_Telescope",
    "https://en.wikipedia.org/wiki/James_Webb_Space_Telescope",
    "https://en.wikipedia.org/wiki/Space_Shuttle",
    "https://en.wikipedia.org/wiki/Artemis_program",
    "https://en.wikipedia.org/wiki/Skylab",
    "https://en.wikipedia.org/wiki/NASA",
    "https://en.wikipedia.org/wiki/European_Space_Agency",
    "https://en.wikipedia.org/wiki/Ariane_(rocket_family)",
    "https://en.wikipedia.org/wiki/Spitzer_Space_Telescope",
    "https://en.wikipedia.org/wiki/New_Horizons",
    "https://en.wikipedia.org/wiki/Cassini%E2%80%93Huygens",
    "https://en.wikipedia.org/wiki/Curiosity_(rover)",
    "https://en.wikipedia.org/wiki/Perseverance_(rover)",
    "https://en.wikipedia.org/wiki/InSight",
    "https://en.wikipedia.org/wiki/OSIRIS-REx",
    "https://en.wikipedia.org/wiki/Parker_Solar_Probe",
    "https://en.wikipedia.org/wiki/BepiColombo",
    # Book title "Juice (spacecraft)" now redirects here (its old title 404s).
    "https://en.wikipedia.org/wiki/Jupiter_Icy_Moons_Explorer",
    "https://en.wikipedia.org/wiki/Solar_Orbiter",
    # Book title "CHEOPS (satellite)" was renamed to the shorter "CHEOPS".
    "https://en.wikipedia.org/wiki/CHEOPS",
    "https://en.wikipedia.org/wiki/Gaia_(spacecraft)",
]

#: Headings whose section (and everything after it) is discarded.
STRIP_SECTIONS: tuple[str, ...] = ("References", "Bibliography", "External links", "See also")

#: Container holding the article body in Wikipedia's HTML.
ARTICLE_CONTAINER = "mw-parser-output"

#: Citation markers: the book's ``[1]`` and current Wikipedia's ``[ 1 ]``.
_CITATION_RE = re.compile(r"\[\s*\d+\s*\]")

#: Where ``collect`` writes by default (git-ignored, like the rest of ``data/raw``).
DEFAULT_OUTPUT = Path("data/raw/llm.md")

#: Wikipedia asks clients to identify themselves.
_HEADERS = {"User-Agent": "my_rag/0.1 (study project; RAG-Driven-Generative-AI chapter 2)"}


def clean_text(content: str) -> str:
    """Remove bracketed citation markers such as ``[1]`` / ``[ 1 ]``."""
    return _CITATION_RE.sub("", content)


def _top_level_block(container: Tag, node: Tag) -> Tag:
    """Return the ancestor of ``node`` that is a direct child of ``container``."""
    while node.parent is not None and node.parent is not container:
        node = node.parent
    return node


def strip_sections(container: Tag) -> None:
    """Delete the trailing reference/navigation sections from ``container`` in place."""
    for title in STRIP_SECTIONS:
        # Wikipedia heading ids use underscores ("See_also" for "See also").
        heading = container.find(id=title.replace(" ", "_")) or container.find(id=title)
        if heading is None:
            continue
        # A heading starts a section that runs to the next one, so dropping the
        # section's top-level block and its later siblings removes the whole tail.
        section = _top_level_block(container, heading)
        for sibling in section.find_next_siblings():
            sibling.decompose()
        section.decompose()


def _article_title(soup: BeautifulSoup) -> str:
    """Return the page's display title (the ``<h1 id="firstHeading">`` text)."""
    heading = soup.find(id="firstHeading")
    if heading is None:
        return ""
    return re.sub(r"\s+", " ", heading.get_text(separator=" ")).strip()


def parse_article(html: str) -> tuple[str, str]:
    """Return ``(title, body)`` for a Wikipedia HTML page, with noise stripped."""
    soup = BeautifulSoup(html, "html.parser")
    containers = soup.find_all("div", {"class": ARTICLE_CONTAINER})
    if not containers:
        raise ValueError(f"no article body found (missing {ARTICLE_CONTAINER!r} container)")
    # Some pages carry a small extra mw-parser-output stub (e.g. a floating box);
    # the article body is the largest one.
    container = max(containers, key=lambda element: len(element.get_text()))
    strip_sections(container)
    return _article_title(soup), clean_text(container.get_text(separator=" ", strip=True))


def extract_text(html: str) -> str:
    """Extract the cleaned article body text from a Wikipedia HTML page."""
    return parse_article(html)[1]


@dataclass(frozen=True)
class Article:
    """A fetched Wikipedia article, ready to be written as Markdown."""

    url: str
    title: str
    text: str

    def to_markdown(self) -> str:
        """Render the article as a Markdown section (one ``#`` heading + body)."""
        heading = self.title or self.url
        return f"# {heading}\n\n{self.text}\n"


def fetch_article(
    url: str,
    session: Optional[requests.Session] = None,
    *,
    timeout: float = 30.0,
) -> Article:
    """Download ``url`` and return its cleaned title and body."""
    client = session or requests
    response = client.get(url, headers=_HEADERS, timeout=timeout)
    response.raise_for_status()
    title, text = parse_article(response.text)
    return Article(url=url, title=title, text=text)


@dataclass(frozen=True)
class CollectionResult:
    """Outcome of :func:`collect`: where the corpus landed and what failed."""

    path: Path
    collected: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


def collect(
    urls: Sequence[str] = WIKI_URLS,
    output: Path = DEFAULT_OUTPUT,
    *,
    session: Optional[requests.Session] = None,
    timeout: float = 30.0,
    progress: Optional[Callable[[str], None]] = None,
) -> CollectionResult:
    """Fetch every URL and write the articles to ``output`` as Markdown.

    Each article becomes a ``#`` heading followed by its cleaned body. A single
    unreachable or unparsable page must not discard the rest of a run, so
    failures are recorded in :attr:`CollectionResult.failed` rather than raised.
    ``progress``, when given, receives a line before each download and for each
    failure.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = CollectionResult(path=output)
    total = len(urls)
    with output.open("w", encoding="utf-8") as file:
        for index, url in enumerate(urls, start=1):
            if progress is not None:
                progress(f"[{index}/{total}] {url}")
            try:
                file.write(fetch_article(url, session, timeout=timeout).to_markdown() + "\n")
            except (requests.RequestException, ValueError) as exc:
                result.failed.append((url, str(exc)))
                if progress is not None:
                    progress(f"    skipped {url}: {exc}")
            else:
                result.collected.append(url)
    return result
