"""Embedding backends (chapter 2, part 2).

Reimplementation of the book's ``embedding_function`` from
``Chapter02/2_Embeddings_vector_store.ipynb``. The book calls OpenAI's
``text-embedding-3-small``; here the default is a free, lightweight local model
served by Ollama (``all-minilm``, ~45 MB, the same all-MiniLM-L6-v2 the book uses
for evaluation). Ollama exposes an OpenAI-compatible ``/v1/embeddings`` endpoint,
so the same :class:`OpenAICompatibleEmbedding` covers both providers — only the
endpoint and model name change.

Deep Lake calls an embedding function through the LangChain-style
``embed_documents`` / ``embed_query`` methods, so the class implements those;
``__call__`` is kept as a convenience for direct use.
"""
from __future__ import annotations

from typing import Optional, Protocol, Sequence

import numpy as np
from openai import OpenAI

from .config import Settings


class EmbeddingFunction(Protocol):
    """The embedding interface Deep Lake expects (LangChain-style)."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""

    def embed_query(self, text: str) -> list[float]:
        """Return the embedding vector for a single query string."""


class OpenAICompatibleEmbedding:
    """Embedding client for any server exposing the OpenAI embeddings API."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text.

        Mirrors the book's function: a bare string is accepted, and newlines are
        flattened so a chunk maps to a single embedding request line.
        """
        if isinstance(texts, str):
            texts = [texts]
        cleaned = [text.replace("\n", " ") for text in texts]
        response = self._client.embeddings.create(input=cleaned, model=self.model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """Return the embedding vector for a single query string."""
        return self.embed_documents([text])[0]

    def __call__(self, texts: Sequence[str] | str) -> list[list[float]]:
        """Alias for :meth:`embed_documents` (the book's ``embedding_function``)."""
        return self.embed_documents(texts)


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    """Cosine of the angle between two embedding vectors (0.0 if either is all zeros)."""
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm else 0.0


class EmbeddingIndex:
    """In-memory semantic index: embed the records once, rank them by cosine similarity.

    Chapter 4's ``VectorStoreIndex.from_documents`` without a persistent store — a few
    hundred records fit in memory. Satisfies :class:`my_rag.pipeline.Retriever`.
    """

    def __init__(
        self,
        records: Sequence[str],
        embedding_function: EmbeddingFunction,
        *,
        top_k: int = 1,
    ) -> None:
        if not records:
            raise ValueError("cannot index an empty list of records")
        self.records = list(records)
        self.embedding_function = embedding_function
        self.top_k = top_k
        vectors = np.asarray(embedding_function.embed_documents(self.records), dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._unit = vectors / np.where(norms == 0, 1.0, norms)

    def search(self, query: str, *, top_k: Optional[int] = None) -> list[tuple[int, float]]:
        """Return ``(position, score)`` for the ``top_k`` records most similar to ``query``."""
        k = self.top_k if top_k is None else top_k
        vector = np.asarray(self.embedding_function.embed_query(query), dtype=float)
        norm = np.linalg.norm(vector)
        scores = self._unit @ (vector / norm) if norm else np.zeros(len(self.records))
        order = np.argsort(scores)[::-1][:k]
        return [(int(position), float(scores[position])) for position in order]

    def retrieve(self, query: str, *, top_k: Optional[int] = None) -> list[str]:
        """Retriever protocol: the text of the ``top_k`` most similar records."""
        return [self.records[position] for position, _ in self.search(query, top_k=top_k)]


def create_embedding_function(settings: Settings) -> EmbeddingFunction:
    """Build the embedding callable requested by ``settings``.

    Both built-in providers speak the OpenAI embeddings API; the model is
    ``settings.embedding_model`` (env ``MY_RAG_EMBEDDING_MODEL``).
    """
    return OpenAICompatibleEmbedding(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.embedding_model,
    )
