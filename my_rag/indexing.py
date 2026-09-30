"""Index-based semantic search with LlamaIndex + Deep Lake (chapter 3).

Reimplementation of ``Chapter03/Deep_Lake_LlamaIndex_OpenAI_RAG.ipynb``
("Constructing an Index-Based Deep Lake Vector Store for Semantic Search with
LlamaIndex and OpenAI"). The book indexes the drone/UAV corpus with LlamaIndex
and queries it through four index types, reporting the retrieved source nodes:

- **vector** — a Deep Lake vector store behind ``VectorStoreIndex`` (semantic);
- **tree** — a hierarchical ``TreeIndex``;
- **list** — a sequential ``ListIndex`` (summary);
- **keyword** — a ``KeywordTableIndex`` (keyword extraction).

This module keeps that flow but, like the rest of ``my_rag``, points the vector
store at a **local** Deep Lake path (no Activeloop account/token) and generates
with the free local Ollama models instead of OpenAI. The book's OpenAI
integrations (``llama-index-llms-openai``) pin ``openai<3`` and would downgrade
the package used elsewhere in ``my_rag``; ``llama-index-llms-ollama`` /
``llama-index-embeddings-ollama`` match the default provider instead, so chapter 3
is scoped to ``--provider ollama``.

LlamaIndex is imported lazily so light commands do not pay for it.
"""
from __future__ import annotations

import logging
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .config import DEFAULT_INDEX_STORE, Settings

#: Deep Lake's analytics reporter can hang a short-lived CLI (see chapter 2).
os.environ.setdefault("BUGGER_OFF", "1")

#: Deep Lake prints a PyPI update banner on import; irrelevant CLI noise.
warnings.filterwarnings("ignore", message="A newer version of deeplake")

#: Index types ported from the book, in notebook order.
INDEX_KINDS: tuple[str, ...] = ("vector", "tree", "list", "keyword")

#: ``similarity_top_k`` default from the book's notebook.
DEFAULT_TOP_K = 3


def _quiet_deeplake() -> None:
    """Re-apply the ``deeplake`` log level after Deep Lake's import-time setup."""
    logging.getLogger("deeplake").setLevel(logging.ERROR)


def _ollama_base_url(base_url: str) -> str:
    """Ollama's client wants the host root, not the OpenAI-compatible ``/v1``."""
    url = (base_url or "").rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url or "http://localhost:11434"


def _require_ollama(settings: Settings) -> None:
    if settings.provider != "ollama":
        raise ValueError(
            f"chapter 3 (LlamaIndex) supports the local 'ollama' provider only, got {settings.provider!r}; "
            "run with --provider ollama"
        )


class _NumpyEmbeddingStore:
    """Proxy that coerces precomputed embeddings to numpy arrays before ``add``.

    Deep Lake 3.9.x — the last line installable on Python 3.14 — runs
    ``np.can_cast`` on every embedding value, which numpy 2 removed for Python
    scalars. Chapter 2 sidesteps this by handing Deep Lake an
    ``embedding_function`` and letting it compute embeddings internally; the
    LlamaIndex integration passes Python floats directly, which trips the bug.
    Wrapping the inner store's ``add`` keeps chapter 3 working without
    duplicating the integration's append logic.
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def add(self, *args: Any, **kwargs: Any) -> Any:
        embedding = kwargs.get("embedding")
        if embedding is not None:
            kwargs["embedding"] = [
                None if vector is None else np.asarray(vector, dtype=np.float32)
                for vector in embedding
            ]
        return self._store.add(*args, **kwargs)


_DEEP_LAKE_INDEX_STORE = None


def deep_lake_index_store_class():
    """Return a numpy-safe :class:`DeepLakeVectorStore` subclass (built lazily)."""
    global _DEEP_LAKE_INDEX_STORE
    if _DEEP_LAKE_INDEX_STORE is None:
        from llama_index.vector_stores.deeplake import DeepLakeVectorStore

        class NumpySafeDeepLakeVectorStore(DeepLakeVectorStore):
            def add(self, nodes, **kwargs):
                original = self.vectorstore
                self.vectorstore = _NumpyEmbeddingStore(original)
                try:
                    return super().add(nodes, **kwargs)
                finally:
                    self.vectorstore = original

        _DEEP_LAKE_INDEX_STORE = NumpySafeDeepLakeVectorStore
    return _DEEP_LAKE_INDEX_STORE


def build_llm(settings: Settings):
    """Build the LlamaIndex LLM backed by the local Ollama server."""
    from llama_index.llms.ollama import Ollama

    return Ollama(
        model=settings.model,
        base_url=_ollama_base_url(settings.base_url),
        temperature=settings.temperature,
        request_timeout=120.0,
    )


def build_embed_model(settings: Settings):
    """Build the LlamaIndex embedding model backed by the local Ollama server."""
    from llama_index.embeddings.ollama import OllamaEmbedding

    return OllamaEmbedding(
        model_name=settings.embedding_model,
        base_url=_ollama_base_url(settings.base_url),
    )


def _split_markdown(text: str) -> list[tuple[str, str]]:
    """Split the collected corpus into ``(title, body)`` sections by ``#`` heading."""
    sections: list[tuple[str, str]] = []
    title: Optional[str] = None
    buffer: list[str] = []
    for line in text.splitlines():
        if line.startswith("# "):
            if title is not None:
                sections.append((title, "\n".join(buffer).strip()))
            title = line[2:].strip()
            buffer = []
        else:
            buffer.append(line)
    if title is not None:
        sections.append((title, "\n".join(buffer).strip()))
    return [(heading, body) for heading, body in sections if body]


def load_documents(path: Path | str) -> list[Any]:
    """Load the collected corpus as LlamaIndex documents.

    A directory is read with :class:`SimpleDirectoryReader` (one file per
    document, as the book does); the single-file Markdown corpus produced by
    :mod:`my_rag.collection` is split into one document per ``#`` heading.
    """
    from llama_index.core import Document, SimpleDirectoryReader

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"corpus {path} not found; collect it with: python -m my_rag --collect --corpus drone")
    if path.is_dir():
        return SimpleDirectoryReader(input_dir=str(path), recursive=True).load_data()

    documents = [
        Document(text=body, metadata={"title": title, "source": path.name})
        for title, body in _split_markdown(path.read_text(encoding="utf-8"))
    ]
    if not documents:
        raise ValueError(f"corpus {path} produced no documents")
    return documents


def build_index(
    kind: str,
    documents: Sequence[Any],
    settings: Settings,
    *,
    vector_store_path: Path | str = DEFAULT_INDEX_STORE,
):
    """Build the requested index over ``documents``.

    The LLM/embedding are installed into LlamaIndex's global ``Settings`` so the
    index and its query engine (built later) share them.
    """
    from llama_index.core import (
        KeywordTableIndex,
        ListIndex,
        Settings as LlamaIndexSettings,
        StorageContext,
        TreeIndex,
        VectorStoreIndex,
    )

    if kind not in INDEX_KINDS:
        raise ValueError(f"unknown index type {kind!r}; available: {', '.join(INDEX_KINDS)}")
    _require_ollama(settings)

    LlamaIndexSettings.llm = build_llm(settings)
    LlamaIndexSettings.embed_model = build_embed_model(settings)

    if kind == "vector":
        _quiet_deeplake()
        # overwrite=True keeps re-runs idempotent instead of appending, as --embed does.
        vector_store = deep_lake_index_store_class()(
            dataset_path=str(vector_store_path), overwrite=True, verbose=False
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    if kind == "tree":
        return TreeIndex.from_documents(documents)
    if kind == "list":
        return ListIndex.from_documents(documents)
    return KeywordTableIndex.from_documents(documents)


@dataclass(frozen=True)
class Source:
    """One node the index returned for a query, with its similarity score."""

    node_id: str
    score: Optional[float]
    text: str


@dataclass(frozen=True)
class IndexAnswer:
    """Outcome of :meth:`IndexQueryEngine.query`."""

    response: str
    sources: list[Source]
    elapsed: float


class IndexQueryEngine:
    """Query one index type, satisfying the chapter 1 :class:`Retriever` protocol.

    The index is built lazily on first use, so constructing the engine is cheap.
    """

    def __init__(
        self,
        kind: str,
        documents: Sequence[Any],
        settings: Settings,
        *,
        vector_store_path: Path | str = DEFAULT_INDEX_STORE,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if kind not in INDEX_KINDS:
            raise ValueError(f"unknown index type {kind!r}; available: {', '.join(INDEX_KINDS)}")
        self.kind = kind
        self.documents = list(documents)
        self.settings = settings
        self.vector_store_path = Path(vector_store_path)
        self.top_k = top_k
        self._index = None
        self._query_engine = None

    @property
    def index(self):
        """The built index (constructed on first access)."""
        if self._index is None:
            self._index = build_index(
                self.kind, self.documents, self.settings, vector_store_path=self.vector_store_path
            )
        return self._index

    def _engine(self):
        if self._query_engine is None:
            self._query_engine = self.index.as_query_engine(similarity_top_k=self.top_k)
        return self._query_engine

    def query(self, question: str) -> IndexAnswer:
        """Run the book's index query and return the answer plus its sources."""
        start = time.perf_counter()
        response = self._engine().query(question)
        elapsed = time.perf_counter() - start
        sources = [
            Source(
                node_id=str(getattr(node_with_score.node, "node_id", "")),
                score=None if node_with_score.score is None else float(node_with_score.score),
                text=node_with_score.node.get_content(),
            )
            for node_with_score in getattr(response, "source_nodes", []) or []
        ]
        return IndexAnswer(response=str(response), sources=sources, elapsed=elapsed)

    def retrieve(self, query: str, *, top_k: Optional[int] = None) -> list[str]:
        """Retriever protocol: the text of the ``top_k`` nodes the index returns."""
        retriever = self.index.as_retriever(similarity_top_k=top_k or self.top_k)
        nodes = retriever.retrieve(query)
        return [node_with_score.node.get_content() for node_with_score in nodes]
