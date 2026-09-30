"""Embedding vector store (chapter 2, part 2).

Reimplementation of ``Chapter02/2_Embeddings_vector_store.ipynb`` ("Embedding-
Based Retrieval with Activeloop and OpenAI"). The book chunks ``llm.txt`` into
1000-character pieces, embeds each chunk, and stores the vectors in an Activeloop
Deep Lake ``VectorStore``; retrieval then embeds the query and searches the
store. This module keeps that flow but points the store at a **local** Deep Lake
path (no Activeloop account/token needed) and embeds with the free local model
configured in :mod:`my_rag.embeddings`.

Deep Lake is imported lazily so that light commands (``--collect``, plain
generation) do not pay for it.
"""
from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import CHUNK_SIZE, DEFAULT_VECTOR_STORE
from .embeddings import EmbeddingFunction

#: Deep Lake's opt-out for its humbug analytics reporter. Left on, the reporter
#: posts from a non-daemon thread at interpreter exit and can hang a short-lived
#: CLI; opting out is humbug's documented switch. ``setdefault`` keeps an
#: explicit BUGGER_OFF from the user (e.g. BUGGER_OFF=0 to re-enable).
os.environ.setdefault("BUGGER_OFF", "1")

#: Deep Lake prints a PyPI update banner on import; irrelevant noise for a CLI run.
warnings.filterwarnings("ignore", message="A newer version of deeplake")


def _load_vector_store_class():
    """Import Deep Lake lazily and quiet its stdout logging.

    Deep Lake's ``configure_logger`` runs on import and (re)sets the ``deeplake``
    logger to INFO with a stdout handler, so the level must be re-applied *after*
    the import. This hides chatty lines such as "already exists, loading from the
    storage"; errors still surface.
    """
    from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore

    logging.getLogger("deeplake").setLevel(logging.ERROR)
    return VectorStore


@dataclass(frozen=True)
class SearchHit:
    """One vector-store match: the chunk, its similarity score, and metadata."""

    text: str
    score: float
    metadata: dict

    def __str__(self) -> str:  # so the pipeline's --show-context prints the chunk
        return self.text


@dataclass(frozen=True)
class BuildResult:
    """Outcome of :meth:`DeepLakeVectorStore.build`."""

    path: Path
    chunks: int
    source: Path


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split ``text`` into fixed-size character chunks, as the book does."""
    return [text[index : index + size] for index in range(0, len(text), size)]


class DeepLakeVectorStore:
    """A Deep Lake vector store bound to an embedding function.

    Satisfies :class:`my_rag.pipeline.Retriever`, so it can be handed straight to
    :class:`my_rag.pipeline.RAGPipeline` for augmented generation.
    """

    def __init__(
        self,
        path: Path | str = DEFAULT_VECTOR_STORE,
        embedding_function: Optional[EmbeddingFunction] = None,
        *,
        chunk_size: int = CHUNK_SIZE,
    ) -> None:
        self.path = Path(path)
        self.embedding_function = embedding_function
        self.chunk_size = chunk_size
        self._store = None

    def _open(self):
        """Open (creating if needed) the underlying Deep Lake vector store."""
        if self._store is None:
            vector_store_class = _load_vector_store_class()
            self._store = vector_store_class(path=str(self.path), verbose=False)
        return self._store

    def build(self, corpus_path: Path | str) -> BuildResult:
        """Embed ``corpus_path`` chunk by chunk into a fresh store at ``self.path``."""
        vector_store_class = _load_vector_store_class()

        corpus_path = Path(corpus_path)
        chunks = chunk_text(corpus_path.read_text(encoding="utf-8"), self.chunk_size)
        chunks = [chunk for chunk in chunks if chunk.strip()]
        if not chunks:
            raise ValueError(f"corpus {corpus_path} produced no chunks to embed")
        # overwrite=True makes re-running --embed idempotent instead of appending.
        self._store = vector_store_class(path=str(self.path), overwrite=True, verbose=False)
        self._store.add(
            text=chunks,
            embedding_function=self.embedding_function,
            embedding_data=chunks,
            metadata=[{"source": corpus_path.name}] * len(chunks),
        )
        return BuildResult(path=self.path, chunks=len(chunks), source=corpus_path)

    def search(self, query: str, *, top_k: int = 1) -> list[SearchHit]:
        """Return the ``top_k`` chunks most similar to ``query``."""
        result = self._open().search(
            embedding_data=query, embedding_function=self.embedding_function, k=top_k
        )
        return [
            SearchHit(text=text, score=float(score), metadata=metadata)
            for text, score, metadata in zip(result["text"], result["score"], result["metadata"])
        ]

    def retrieve(self, query: str, *, top_k: int = 1) -> list[str]:
        """Retriever protocol: the text of the ``top_k`` most similar chunks."""
        return [hit.text for hit in self.search(query, top_k=top_k)]


def open_vector_store(
    path: Path | str = DEFAULT_VECTOR_STORE,
    embedding_function: Optional[EmbeddingFunction] = None,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> DeepLakeVectorStore:
    """Open an existing store, failing clearly when it has not been built yet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"vector store {path} not found; build it with: python -m my_rag --embed")
    return DeepLakeVectorStore(path, embedding_function, chunk_size=chunk_size)
