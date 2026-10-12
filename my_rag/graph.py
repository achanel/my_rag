"""Knowledge-graph RAG over Wikipedia (chapter 7).

Reimplementation of ``Chapter07/Knowledge_Graph__Deep_Lake_LlamaIndex_OpenAI_RAG.ipynb``
("Constructing a Knowledge Graph-Based Deep Lake Vector Store for Semantic Search
with LlamaIndex and OpenAI"), plus the two notebooks it leans on:

- ``Wikipedia_API.ipynb`` — walk a Wikipedia page's links and keep each linked
  page's summary. Its output is the book's ``Marketing_urls.txt`` citation list;
  :data:`MARKETING_URLS_URL` is that ready-made file, and the summary walk
  (:func:`crawl_summaries`) reproduces it from the API alone.
- ``Tree_2_Graph.ipynb`` — build a directed graph from pairs, style the edges by
  "friendship", draw it. Ported as :func:`build_networkx_graph` /
  :func:`draw_networkx_graph` for the book's toy ``pairs`` example and as the
  ``--visualize`` output for the real graph.
- the KG notebook proper — collect and clean the documents (pipeline 1), index
  them (pipeline 2), then query the index, re-rank the source nodes by a semantic
  cosine, and report metrics (pipeline 3).

LlamaIndex's ``KnowledgeGraphIndex`` extracts ``(subject, predicate, object)``
triplets from every chunk with one LLM call each, so the graph of the book's full
100-page corpus is hours on a local model; :data:`DEFAULT_LIMIT` keeps a run
short and ``--limit`` scales it up.

Substitutions, consistent with the rest of ``my_rag``:

- **Local Ollama instead of GPT-4o**, through the same LLM/embedding builders as
  chapter 3 (the OpenAI LlamaIndex integrations pin ``openai<3``).
- **A local index folder instead of ``hub://denis76/marketing01``.** The book
  writes both the Deep Lake vector store and the dataset to Activeloop; the graph
  index persists to :data:`~my_rag.config.DEFAULT_GRAPH_STORE` (no token/network),
  which also lets ``--rag graph`` reload it instead of re-extracting triplets.
- **The ready-made ``Marketing_urls.txt`` from the book's GitHub**, not a Colab
  copy of the 100 URLs; ``--crawl`` re-derives the list from the Wikipedia API.
- **Chapter 2's ``clean_text``/parser** (:mod:`my_rag.collection`) instead of the
  notebook's second copy of the fetcher.
- **pyvis instead of the notebook's inline HTML cell** for the graph display, so
  the graph opens as a file in a browser.
- **The TF-IDF/numpy TF-IDF cosine instead of ``sentence-transformers``** (both
  the book's re-ranking and its metrics cell call ``all-MiniLM-L6-v2``): the
  re-ranking asks the configured embedding model, the metrics reuse chapter 5's
  :func:`~my_rag.adaptive.tfidf_similarity`. Like chapter 3, the book's separate
  evaluation dependency is not ported.

Not ported: the Colab scaffolding (Drive mount, ``api_key.txt``/``activeloop.txt``
key files, the ``/etc/resolv.conf`` DNS cell, the Pillow-version restart, the
``!pip install`` pins), the notebook's ``display()``/``print`` cells, and
``deeplake.load(...).summary()`` (a DataFrame dump of the store chapter 2/3 already
cover) — :func:`GraphIndex.stats` reports the graph's own shape instead.
"""
from __future__ import annotations

import itertools
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import requests

from .collection import Article, clean_text, fetch_article, parse_article
from .config import (
    DEFAULT_GRAPH_HTML_DIR,
    DEFAULT_GRAPH_LIMIT,
    DEFAULT_GRAPH_STORE,
    Settings,
)
from .embeddings import create_embedding_function
from .llm import LLM

#: The book's citation list for its ``graph_name = "Marketing"`` scenario: 100
#: Wikipedia URLs, downloaded to ``data/raw/marketing_urls.txt`` on collect.
MARKETING_URLS_URL = (
    "https://raw.githubusercontent.com/Denis2054/RAG-Driven-Generative-AI/"
    "main/Chapter07/citations/Marketing_urls.txt"
)

#: Wikipedia API endpoint used to re-derive the list with ``--crawl``.
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

#: Where ``--collect --corpus marketing`` writes the corpus (one ``#`` heading per
#: page, like chapter 2's corpus) and the URL list it was built from.
DEFAULT_MARKETING_OUTPUT = Path("data/raw/marketing.md")
DEFAULT_URLS_OUTPUT = Path("data/raw/marketing_urls.txt")

#: Link prefixes kept by the citation walk — Wikipedia articles only, so page
#: furniture (categories, files, the API itself) never enters the graph.
_ARTICLE_PREFIX = "https://en.wikipedia.org/wiki/"
_NS_MAIN = 0

#: ``sentence-transformers``' truncation point in the book's metrics cell; all-minilm
#: sees the same first 512 tokens, so the re-ranking cosine matches the book's setup.
_MAX_WORDS = 512

#: Default ``similarity_top_k`` and ``temperature`` of the book's graph query engine.
DEFAULT_TOP_K = 3
DEFAULT_TEMPERATURE = 0.1

#: Sample size and rating thresholds of the notebook's final metrics cells.
METRIC_RECORDS = 10
METRIC_CEILING = 0.90

#: The book's Tree_2_Graph example, kept runnable via ``--visualize --pairs``.
TOY_PAIRS: list[tuple[str, str]] = [
    ("a", "b"), ("b", "e"), ("e", "m"), ("m", "p"), ("a", "z"), ("b", "q"),
]
TOY_FRIENDS: set[tuple[str, str]] = {("a", "b"), ("b", "e"), ("e", "m"), ("m", "p")}
TOY_LAYOUTS = ("spring", "planar", "spiral", "shell")

#: A fetcher takes a URL and returns a cleaned :class:`~my_rag.collection.Article`.
Fetcher = Callable[[str], Article]


# --------------------------------------------------------------------------- #
# Pipeline 1 — collecting and preparing the documents
# --------------------------------------------------------------------------- #
def parse_urls(text: str) -> list[str]:
    """Pull the Wikipedia URLs out of a citation file, one per line, deduplicated."""
    seen: dict[str, None] = {}
    for line in text.splitlines():
        url = line.strip()
        if url.startswith(_ARTICLE_PREFIX):
            seen.setdefault(url, None)
    return list(seen)


def collect_urls(
    output: Path | str = DEFAULT_URLS_OUTPUT,
    *,
    session: Optional[requests.Session] = None,
    timeout: float = 30.0,
) -> list[str]:
    """Download the book's ready-made Marketing citation list to ``output``."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    client = session or requests
    response = client.get(MARKETING_URLS_URL, timeout=timeout)
    response.raise_for_status()
    urls = parse_urls(response.text)
    if not urls:
        raise ValueError(f"{MARKETING_URLS_URL} contained no Wikipedia URLs")
    output.write_text("\n".join(urls) + "\n", encoding="utf-8")
    return urls


def _article_links(page: dict) -> list[str]:
    """The page's linked articles, in Wikipedia's own order (namespace 0 titles)."""
    return [link["title"] for link in page.get("links", []) if link.get("ns", _NS_MAIN) == _NS_MAIN]


def crawl_summaries(
    topic: str = "Marketing",
    limit: int = 100,
    *,
    summary_only: bool = True,
) -> list[Article]:
    """Walk ``topic``'s links and return the root page plus one entry per linked page.

    Reproduces the book's ``Wikipedia_API.ipynb`` cells without ``wikipediaapi`` or
    nltk: ``action=query&prop=extracts&exintro`` returns exactly the summary text
    the notebook prints with its token count. ``summary_only=False`` fetches the
    full plaintext instead, which is what the corpus wants.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")

    def fetch_page(title: str, *, with_links: bool = False) -> dict:
        response = requests.get(
            WIKIPEDIA_API,
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "info|extracts|links" if with_links else "info|extracts",
                "inprop": "url",
                "plnamespace": 0,
                "pllimit": "max",
                "exintro": 1 if summary_only else None,
                "explaintext": 1,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), None)
        if not page or "missing" in page:
            raise ValueError(f"Wikipedia page {title!r} not found")
        return page

    root = fetch_page(topic, with_links=True)
    titles = _article_links(root)[:limit]
    articles = [
        Article(
            url=root.get("fullurl") or f"{_ARTICLE_PREFIX}{topic}",
            title=root["title"],
            text=clean_text(root.get("extract", "")),
        )
    ]
    for title in titles:
        page = fetch_page(title)
        articles.append(
            Article(
                url=page.get("fullurl") or f"{_ARTICLE_PREFIX}{title.replace(' ', '_')}",
                title=page["title"],
                text=clean_text(page.get("extract", "")),
            )
        )
    return [article for article in articles if article.text.strip()]


def collect_marketing(
    output: Path | str = DEFAULT_MARKETING_OUTPUT,
    *,
    limit: int = DEFAULT_GRAPH_LIMIT,
    urls_file: Path | str = DEFAULT_URLS_OUTPUT,
    fetcher: Fetcher = fetch_article,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[Path, list[str], list[tuple[str, str]]]:
    """Fetch the first ``limit`` Marketing URLs into one Markdown corpus.

    The book writes one ``.txt`` per URL and reads the folder with
    ``SimpleDirectoryReader``; :func:`~my_rag.collection.collect` already turns a
    URL list into a single Markdown file (one ``#`` heading per page), which
    chapter 3's loader splits back into documents — the same shape with one file
    to clean up. Unreachable pages are reported rather than aborting the run.
    Returns ``(corpus path, fetched URLs, failures)``.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    urls = collect_urls(urls_file)[:limit]
    collected: list[str] = []
    failed: list[tuple[str, str]] = []
    with output.open("w", encoding="utf-8") as file:
        for index, url in enumerate(urls, start=1):
            if progress is not None:
                progress(f"[{index}/{len(urls)}] {url}")
            try:
                file.write(fetcher(url).to_markdown() + "\n")
            except (requests.RequestException, ValueError) as exc:
                failed.append((url, str(exc)))
                if progress is not None:
                    progress(f"    skipped {url}: {exc}")
            else:
                collected.append(url)
    return output, collected, failed


def load_documents(corpus: Path | str, *, limit: Optional[int] = None) -> list[Any]:
    """Load the chapter 7 corpus as LlamaIndex documents (one per ``#`` heading).

    Chapter 3's :func:`~my_rag.indexing.load_documents` already splits the
    collected Markdown this way; this is the same loader with the chapter 7 corpus
    as its default and an optional cap on how many pages the graph is built from.
    """
    from .indexing import load_documents as load  # lazy: LlamaIndex is heavy

    documents = load(corpus)
    return documents[:limit] if limit else documents


# --------------------------------------------------------------------------- #
# Pipeline 2 — the knowledge-graph index
# --------------------------------------------------------------------------- #
def _ollama_llm(settings: Settings):
    """Chapter 3's LlamaIndex Ollama LLM (Ollama-only, like the rest of LlamaIndex)."""
    from .indexing import build_llm

    return build_llm(settings)


@dataclass(frozen=True)
class GraphStats:
    """The graph's shape: triplets, nodes, edges and their density."""

    triplets: int
    nodes: int
    edges: int

    @property
    def density(self) -> float:
        """Share of the possible directed edges that exist (0 when < 2 nodes)."""
        possible = self.nodes * (self.nodes - 1)
        return self.edges / possible if possible else 0.0

    def __str__(self) -> str:  # so the CLI can print it straight
        return (
            f"{self.triplets} triplets, {self.nodes} nodes, {self.edges} edges "
            f"(density {self.density:.4f})"
        )


def graph_triplets(index) -> list[tuple[str, str, str]]:
    """Every ``(subject, predicate, object)`` triplet the index extracted.

    ``SimpleGraphStore.get_rel_map(depth=1)`` returns, per subject, a list of
    ``[subject, predicate, object]`` triples, so every entry is already one
    triplet. Reading them here is more complete than
    ``KnowledgeGraphIndex.get_networkx_graph``, which caps the graph with its
    ``limit`` and drops any path whose next node is missing from the index's
    subject table (an object that never appeared as a subject itself).
    """
    triplets: list[tuple[str, str, str]] = []
    subjects = list(index.index_struct.table.keys())
    rel_map = index.graph_store.get_rel_map(subjects, depth=1, limit=1_000_000)
    for relations in rel_map.values():
        for relation in relations:
            if len(relation) == 3:
                triplet = tuple(str(part) for part in relation)
                if triplet not in triplets:
                    triplets.append(triplet)
    return triplets


def _graph_from_triplets(triplets: Sequence[tuple[str, str, str]]):
    """Build the directed networkx graph the book draws (edges labelled by relation)."""
    import networkx as nx

    graph = nx.DiGraph()
    for subject, relation, obj in triplets:
        graph.add_edge(subject, obj, label=relation, title=relation)
    return graph


def build_networkx_graph(pairs: Sequence[tuple[str, str]]):
    """The book's ``Tree_2_Graph`` tree builder: pairs → directed graph + root.

    The root is the single node with no incoming edge (``None`` when the pairs have
    several roots or none, as the notebook warns).
    """
    import networkx as nx

    graph = nx.DiGraph()
    for first, second in pairs:
        graph.add_edge(first, second)
    roots = [node for node in graph.nodes() if graph.in_degree(node) == 0]
    return graph, (roots[0] if len(roots) == 1 else None)


def draw_networkx_graph(
    graph,
    *,
    output: Path | str,
    layout: str = "spring",
    friends: Optional[set[tuple[str, str]]] = None,
    title: str = "",
) -> Path:
    """The book's tree drawing: solid edges for friend pairs, dashed for the rest.

    Uses ``networkx`` to lay the graph out and write an SVG (no display); the file
    is what the notebook's ``plt.show()`` cell would have shown.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    layouts = {
        "spring": nx.spring_layout,
        "planar": nx.planar_layout,
        "spiral": nx.spiral_layout,
        "shell": nx.shell_layout,
    }
    if layout not in layouts:
        raise ValueError(f"unknown layout {layout!r}; available: {', '.join(layouts)}")

    friends = friends or set()
    planned = layouts[layout](graph)
    figure, axes = plt.subplots(figsize=(10, 10))
    for (first, second) in graph.edges():
        nx.draw_networkx_edges(
            graph,
            planned,
            edgelist=[(first, second)],
            edge_color="black",
            style="solid" if (first, second) in friends else "dashed",
            width=2,
            ax=axes,
        )
    nx.draw_networkx_nodes(
        graph, planned, node_color="lightgray", node_size=1000, edgecolors="black", ax=axes
    )
    nx.draw_networkx_labels(graph, planned, font_size=10, ax=axes)
    axes.set_title(title or f"Visual Representation of the Tree ({layout} layout)")
    axes.axis("off")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg" if output.suffix == ".svg" else "png", bbox_inches="tight")
    plt.close(figure)
    return output


@dataclass(frozen=True)
class GraphBuild:
    """Outcome of :meth:`GraphIndex.build`."""

    documents: int
    chunks: int
    stats: GraphStats
    elapsed: float


class GraphIndex:
    """A ``KnowledgeGraphIndex`` over the chapter 7 corpus (the book's pipeline 2).

    The index is built lazily, so ``--rag graph`` on an already-persisted store
    skips the triplet extraction entirely.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        store_path: Path | str = DEFAULT_GRAPH_STORE,
        llm: Optional[LLM] = None,
        embed_model: Optional[Any] = None,
        embedding_function: Optional[Any] = None,
        max_triplets_per_chunk: int = 2,
        include_embeddings: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        embed_max_retries: int = 3,
    ) -> None:
        self.settings = settings
        self.store_path = Path(store_path)
        self.llm = llm
        self.embed_model = embed_model
        self.embedding_function = embedding_function
        self.max_triplets_per_chunk = max_triplets_per_chunk
        self.include_embeddings = include_embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_max_retries = embed_max_retries
        self._index = None

    def _install_models(self) -> None:
        """Put the LLM/embedding into LlamaIndex's global settings (chapter 3 style)."""
        from llama_index.core import Settings as LlamaIndexSettings

        LlamaIndexSettings.llm = self._llm()
        LlamaIndexSettings.embed_model = self._resolve_embed_model()

    def _resolve_embed_model(self):
        """The injected embedding model, or the Ollama one with a bounded retry envelope.

        Chapter 2/3's builders pass no retries, so LlamaIndex's default (6 tries with
        backoff) turns one flaky embedding into a long stall — the notebook's
        ``batch_upsert`` caps them the same way.
        """
        if self.embed_model is not None:
            return self.embed_model

        from llama_index.embeddings.ollama import OllamaEmbedding

        from .indexing import _ollama_base_url

        return OllamaEmbedding(
            model_name=self.settings.embedding_model,
            base_url=_ollama_base_url(self.settings.base_url),
            max_retries=self.embed_max_retries,
        )

    def _resolve_embedding_function(self):
        """The embedding function the re-ranking uses (injectable for offline runs)."""
        if self.embedding_function is None:
            self.embedding_function = create_embedding_function(self.settings)
        return self.embedding_function

    # -- construction ------------------------------------------------------- #
    def _llm(self):
        if self.llm is None:
            self.llm = _ollama_llm(self.settings)
        return self.llm

    def _require_ollama(self) -> None:
        if self.settings.provider != "ollama":
            raise ValueError(
                "chapter 7 (LlamaIndex knowledge graph) supports the local 'ollama' provider "
                f"only, got {self.settings.provider!r}; run with --provider ollama"
            )

    def _split(self, documents: Sequence[Any]) -> list[Any]:
        from llama_index.core.node_parser import SentenceSplitter

        splitter = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return splitter.get_nodes_from_documents(list(documents))

    def build(
        self,
        documents: Sequence[Any],
        *,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> GraphBuild:
        """Extract triplets from every chunk, embed them, and persist the index."""
        from llama_index.core import KnowledgeGraphIndex, StorageContext
        from llama_index.core.graph_stores import SimpleGraphStore

        self._require_ollama()
        self._install_models()
        nodes = self._split(documents)
        if not nodes:
            raise ValueError("the corpus produced no chunks to index")
        start = time.perf_counter()
        graph_store = SimpleGraphStore()
        self._index = KnowledgeGraphIndex(
            nodes=nodes[:0],  # start empty; chunks are added below so progress is visible
            graph_store=graph_store,
            max_triplets_per_chunk=self.max_triplets_per_chunk,
            include_embeddings=self.include_embeddings,
            show_progress=False,
        )
        for position, node in enumerate(nodes, start=1):
            self._index.insert_nodes([node])
            if progress is not None:
                progress(position, len(nodes))
        elapsed = time.perf_counter() - start
        self._documents = list(documents)
        self._chunks = len(nodes)
        self.persist()
        return GraphBuild(
            documents=len(documents),
            chunks=len(nodes),
            stats=self.stats,
            elapsed=elapsed,
        )

    # -- persistence -------------------------------------------------------- #
    @property
    def exists(self) -> bool:
        """Whether a persisted index is on disk."""
        return (self.store_path / "index_store.json").is_file()

    def persist(self) -> Path:
        """Write the index (triplets, nodes, embeddings) to :attr:`store_path`."""
        if self._index is None:
            raise ValueError("nothing to persist; build or load the index first")
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._index.storage_context.persist(persist_dir=str(self.store_path))
        return self.store_path

    def load(self) -> None:
        """Reopen the persisted index, skipping triplet extraction."""
        from llama_index.core import StorageContext, load_index_from_storage

        if not self.exists:
            raise FileNotFoundError(
                f"knowledge graph {self.store_path} not found; build it with: "
                "python -m my_rag --graph"
            )
        self._require_ollama()
        self._install_models()
        context = StorageContext.from_defaults(persist_dir=str(self.store_path))
        self._index = load_index_from_storage(context)

    @property
    def index(self):
        """The live index: built by :meth:`build`, loaded by :meth:`load`.

        Falls back to the persisted store so a query after ``--graph`` (or a fresh
        process) does not re-extract triplets.
        """
        if self._index is None:
            self.load()
        return self._index

    # -- reporting ---------------------------------------------------------- #
    @property
    def graph(self):
        """The directed networkx graph of the extracted triplets."""
        return _graph_from_triplets(self.triplets)

    @property
    def triplets(self) -> list[tuple[str, str, str]]:
        """All triplets the index holds."""
        return graph_triplets(self.index)

    @property
    def stats(self) -> GraphStats:
        """Triplet/node/edge counts of the live index."""
        triplets = self.triplets
        network = _graph_from_triplets(triplets)
        return GraphStats(
            triplets=len(triplets),
            nodes=network.number_of_nodes(),
            edges=network.number_of_edges(),
        )

    def query(self, question: str, *, top_k: int = DEFAULT_TOP_K) -> "GraphAnswer":
        """Answer ``question`` from the graph, re-ranked, with the book's metrics."""
        engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            temperature=DEFAULT_TEMPERATURE,
            num_output=1024,
        )
        start = time.perf_counter()
        response = engine.query(question)
        elapsed = time.perf_counter() - start
        sources = [
            GraphSource(
                node_id=str(getattr(node.node, "node_id", "")),
                score=None if node.score is None else float(node.score),
                text=node.node.get_content(),
            )
            for node in getattr(response, "source_nodes", []) or []
        ]
        ranking = rerank(question, sources, self._resolve_embedding_function())
        return GraphAnswer(
            question=question,
            response=str(response),
            sources=sources,
            ranking=ranking,
            triplets=len(self.triplets),
            elapsed=elapsed,
        )

    def visualize(
        self,
        path: Path | str,
        *,
        max_nodes: int = 150,
        progress: Optional[Callable[[str], None]] = None,
    ) -> Path:
        """Write the book's pyvis graph (lightgray nodes, black edges) to ``path``."""
        from pyvis.network import Network

        graph = self.graph
        subset = _largest_component(graph, max_nodes=max_nodes) if max_nodes else graph
        if progress is not None:
            progress(
                f"drawing {subset.number_of_nodes()} of {graph.number_of_nodes()} nodes "
                f"({subset.number_of_edges()} of {graph.number_of_edges()} edges)"
            )
        network = Network(notebook=False, cdn_resources="in_line", directed=True)
        network.from_nx(subset)
        for node in network.nodes:
            node["color"] = "lightgray"
            node["size"] = 10
        for edge in network.edges:
            edge["color"] = "black"
            edge["width"] = 1
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        network.write_html(str(path), notebook=False)
        return path


def _largest_component(graph, *, max_nodes: int):
    """Keep the biggest weakly-connected component, trimmed to ``max_nodes``.

    A full-corpus graph is mostly one hub per page plus their out-links; pyvis
    renders hundreds of disconnected dots poorly and slowly, so the drawing shows
    the neighbourhood that actually has structure.
    """
    import networkx as nx

    undirected = graph.to_undirected()
    components = sorted(nx.connected_components(undirected), key=len, reverse=True)
    if not components:
        return graph
    keep = set(components[0])
    if len(keep) > max_nodes:
        keep = set(itertools.islice(keep, max_nodes))
    return graph.subgraph(keep).copy()


# --------------------------------------------------------------------------- #
# Pipeline 3 — query, re-ranking, metrics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GraphSource:
    """One source node the graph retriever returned, with its similarity score."""

    node_id: str
    score: Optional[float]
    text: str


@dataclass(frozen=True)
class RankedSource:
    """A source node scored against a text by cosine similarity (the book's re-ranking)."""

    rank: int
    similarity: float
    text: str


@dataclass(frozen=True)
class Ranking:
    """The re-ranking result: the best node, or none when there were no sources."""

    best_rank: Optional[int]
    best_score: float
    best_text: str
    scores: list[RankedSource] = field(default_factory=list)


def cosine_scores(target: str, texts: Sequence[str], embedding_function) -> list[float]:
    """Cosine similarity of ``target`` against each of ``texts``.

    The book's ``calculate_cosine_similarity_with_embeddings`` — same all-MiniLM
    family model, so the numbers line up with its printed scores.
    """
    if not texts:
        return []
    vectors = np.asarray(
        [
            embedding_function.embed_query(target),
            *embedding_function.embed_documents([_first_words(text) for text in texts]),
        ],
        dtype=float,
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms == 0, 1.0, norms)
    return [float(score) for score in unit[0] @ unit[1:].T]


def _first_words(text: str, words: int = _MAX_WORDS) -> str:
    """First ``words`` whitespace words — the embedding model's context window."""
    return " ".join(text.split()[:words])


def rerank(query: str, sources: Sequence[GraphSource], embedding_function) -> Ranking:
    """Score each source node against the query and report the best one.

    The notebook's ``best_rank``/``best_score``/``best_text`` loop: index 1 is the
    top-ranked node the graph retriever returned.
    """
    if not sources:
        return Ranking(best_rank=None, best_score=0.0, best_text="")
    scores = cosine_scores(query, [source.text for source in sources], embedding_function)
    ranked = [
        RankedSource(rank=position, similarity=score, text=source.text)
        for position, (source, score) in enumerate(zip(sources, scores), start=1)
    ]
    best = max(ranked, key=lambda item: item.similarity)
    return Ranking(
        best_rank=best.rank, best_score=round(best.similarity, 3), best_text=best.text, scores=ranked
    )


@dataclass(frozen=True)
class GraphAnswer:
    """Outcome of :meth:`GraphIndex.query`: the answer, its sources and the metrics."""

    question: str
    response: str
    sources: list[GraphSource]
    ranking: Ranking
    triplets: int
    elapsed: float

    @property
    def similarity(self) -> float:
        """Cosine of the answer against the question (chapter 5's TF-IDF metric)."""
        from .adaptive import tfidf_similarity

        return tfidf_similarity(self.question, self.response)


@dataclass(frozen=True)
class Metrics:
    """The book's metrics cell: descriptive statistics over the similarity sample."""

    mean: float
    median: float
    std: float
    variance: float
    minimum: float
    maximum: float
    range: float
    q1: float
    q3: float
    iqr: float

    def lines(self) -> list[str]:
        """The notebook's ``print`` lines, one per statistic."""
        return [
            f"Mean: {self.mean:.2f}",
            f"Median: {self.median:.2f}",
            f"Standard Deviation: {self.std:.2f}",
            f"Variance: {self.variance:.2f}",
            f"Minimum: {self.minimum:.2f}",
            f"Maximum: {self.maximum:.2f}",
            f"Range: {self.range:.2f}",
            f"25th Percentile (Q1): {self.q1:.2f}",
            f"75th Percentile (Q3): {self.q3:.2f}",
            f"Interquartile Range (IQR): {self.iqr:.2f}",
        ]


def cosine_metrics(scores: Sequence[float]) -> Metrics:
    """Mean/median/dispersion of a similarity sample, as the book's metrics cell."""
    if not scores:
        raise ValueError("cannot compute metrics for an empty sample")
    values = np.asarray(scores, dtype=float)
    q1, q3 = np.percentile(values, 25), np.percentile(values, 75)
    return Metrics(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values)),
        variance=float(np.var(values)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        range=float(np.max(values) - np.min(values)),
        q1=float(q1),
        q3=float(q3),
        iqr=float(q3 - q1),
    )


def judge(score: float) -> int:
    """The book's comparison chart: turn a similarity into its 1–5-quality rating."""
    steps = [(METRIC_CEILING, 5), (0.70, 4), (0.60, 3), (0.50, 2)]
    return next((rating for threshold, rating in steps if score >= threshold), 1)


def comparison_metrics(scores: Sequence[float], human: Optional[Sequence[float]] = None) -> "Comparison":
    """Mean cosine against the book's human scores (``rscores``), when they are given.

    The notebook scatters the ten cosine scores against its hand-entered human
    scores (0.75, 0.5, 0.8, …) and never fits anything to them, so the comparison
    here is the two means — enough to see whether the metric tracks human judgement.
    Without ``human`` the human side is reported as the rubric of :func:`judge`,
    clearly labelled, rather than pretending to be the notebook's hand labels.
    """
    if not scores:
        raise ValueError("cannot compare an empty sample")
    if human is not None:
        if len(human) != len(scores):
            raise ValueError(f"got {len(human)} human scores for {len(scores)} samples")
        human_scores = [float(value) for value in human]
        source = "recorded"
    else:
        human_scores = [judge(score) / 5 for score in scores]
        source = "rubric"
    return Comparison(
        samples=len(scores),
        mean_cosine=float(np.mean(scores)),
        mean_human=float(np.mean(human_scores)),
        agreement=float(np.mean([abs(cosine - value) <= 0.10 for cosine, value in zip(scores, human_scores)])),
        source=source,
    )


@dataclass(frozen=True)
class Comparison:
    """Cosine vs the human/ rubric score, aggregated (see :func:`comparison_metrics`)."""

    samples: int
    mean_cosine: float
    mean_human: float
    agreement: float
    source: str = "rubric"


#: The book's sample questions from the Marketing scenario (``Wikipedia_API.ipynb``
#: and the KG notebook's ten metric runs).
QUESTIONS: tuple[str, ...] = (
    "What is the primary goal of marketing for the consumer market?",
    "Which experts are often associated with marketing theory?",
    "How does marketing boost sales?",
    "What is the difference between B2B and B2C?",
    "What are the 4Ps? What do they stand for?",
    "What are the 4Cs? What do they stand for?",
    "What is the difference between the 4Ps and 4Cs?",
    "What commodity programs does the Agricultural Marketing Service (AMS) maintain?",
    "What kind of marketing is Got Milk?",
    "What is an industry trade group, business association, sector association or industry body?",
    "How many members are there in the American Marketing Association (AMA)?",
)


class GraphRAG:
    """The book's pipeline 3 over a :class:`GraphIndex` (and a query engine shortcut)."""

    def __init__(self, index: GraphIndex, *, top_k: int = DEFAULT_TOP_K) -> None:
        self.index = index
        self.top_k = top_k

    def answer(self, question: str) -> GraphAnswer:
        """Query the graph index and re-rank its source nodes."""
        return self.index.query(question, top_k=self.top_k)

    def evaluate(
        self,
        questions: Sequence[str] = QUESTIONS,
        *,
        limit: int = METRIC_RECORDS,
        progress: Optional[Callable[[int, int, str], None]] = None,
        human_feedback: Optional[Sequence[float]] = None,
    ):
        """Run ``limit`` of the book's questions and aggregate the cosine scores.

        Returns ``(answers, metrics, comparison)``. ``human_feedback`` stands in for
        the notebook's hand-entered ``rscores`` (0.75, 0.5, 0.8, ...); without it the
        comparison falls back to the labelled rubric of :func:`judge`.
        """
        answers = []
        for position, question in enumerate(questions[:limit], start=1):
            answers.append(self.answer(question))
            if progress is not None:
                progress(position, min(limit, len(questions)), question)
        scores = [answer.similarity for answer in answers]
        metrics = cosine_metrics(scores)
        return answers, metrics, comparison_metrics(scores, human_feedback)


def write_report(path: Path | str, *, answers: Sequence[GraphAnswer], metrics: Metrics, comparison: Comparison) -> Path:
    """Write the metrics run (questions, answers, scores, statistics) as JSON.

    The notebook keeps all of this in cell outputs; a file makes the numbers
    inspectable and diffable after the run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "answers": [
            {
                "question": answer.question,
                "response": answer.response,
                "similarity": round(answer.similarity, 4),
                "best_rank": answer.ranking.best_rank,
                "best_score": answer.ranking.best_score,
                "sources": [round(source, 4) for source in [item.similarity for item in answer.ranking.scores]],
                "elapsed": round(answer.elapsed, 3),
            }
            for answer in answers
        ],
        "metrics": {line.split(":")[0]: line.split(":")[1].strip() for line in metrics.lines()},
        "comparison": {
            "samples": comparison.samples,
            "mean_cosine": round(comparison.mean_cosine, 4),
            "mean_human": round(comparison.mean_human, 4),
            "agreement": round(comparison.agreement, 4),
            "source": comparison.source,
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def target_html_path(name: str = "Knowledge_graph", directory: Path | str = DEFAULT_GRAPH_HTML_DIR) -> Path:
    """Where a graph drawing goes: ``<directory>/<name>.html``."""
    return Path(directory) / f"{name}.html"


def safe_name(text: str) -> str:
    """Turn a question/topic into a filename-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60] or "graph"


__all__ = [
    "MARKETING_URLS_URL",
    "WIKIPEDIA_API",
    "DEFAULT_MARKETING_OUTPUT",
    "DEFAULT_URLS_OUTPUT",
    "DEFAULT_GRAPH_STORE",
    "DEFAULT_GRAPH_LIMIT",
    "DEFAULT_TOP_K",
    "METRIC_RECORDS",
    "TOY_PAIRS",
    "TOY_FRIENDS",
    "TOY_LAYOUTS",
    "QUESTIONS",
    "parse_urls",
    "collect_urls",
    "collect_marketing",
    "crawl_summaries",
    "load_documents",
    "build_networkx_graph",
    "draw_networkx_graph",
    "graph_triplets",
    "cosine_scores",
    "rerank",
    "cosine_metrics",
    "comparison_metrics",
    "judge",
    "GraphIndex",
    "GraphRAG",
    "GraphAnswer",
    "GraphBuild",
    "GraphSource",
    "GraphStats",
    "RankedSource",
    "Ranking",
    "Metrics",
    "Comparison",
    "write_report",
    "target_html_path",
    "safe_name",
]
