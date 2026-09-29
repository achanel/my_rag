"""Retrieval strategies: Naive, Advanced, and Modular RAG (chapter 1, part 2).

The reference notebook introduces three RAG variants, distinguished by how the
context is retrieved before generation:

- **Naive RAG** — keyword overlap between the query and each record
  (:class:`KeywordRetriever`).
- **Advanced RAG** — TF-IDF cosine similarity: :class:`VectorRetriever` fits a
  vectorizer per query/record pair (the brute-force method of §3.1), while
  :class:`IndexRetriever` queries a once-precomputed TF-IDF matrix (§3.2).
- **Modular RAG** — :class:`ModularRetriever` dispatches to any of the above by
  name, mirroring the book's ``RetrievalComponent(method=...)``.

Every retriever implements the :class:`my_rag.pipeline.Retriever` protocol, so
any of them can be passed to :class:`my_rag.pipeline.RAGPipeline`. The TF-IDF
math is implemented on numpy alone to keep ``my_rag`` on its core dependency
stack (no scikit-learn).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")

#: Minimal English stop-word list (stand-in for sklearn's ``stop_words='english'``).
STOP_WORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by cannot could did do does doing down
    during each few for from further had has have having he her here hers herself
    him himself his how i if in into is it its itself me more most my myself no
    nor not of off on once only or other our ours ourselves out over own same she
    should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what
    when where which while who whom why will with you your yours yourself
    yourselves
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens of length > 1, with stop words removed."""
    return [word for word in _WORD_RE.findall(text.lower()) if len(word) > 1 and word not in STOP_WORDS]


def keywords(text: str) -> set[str]:
    """Raw keyword set (no stop-word filtering), as the book's keyword search does."""
    return set(_WORD_RE.findall(text.lower()))


class TfidfIndex:
    """A fitted TF-IDF model over a fixed corpus.

    Uses sublinear term frequency and smooth inverse document frequency, then
    L2-normalizes rows so that a plain dot product equals cosine similarity.
    """

    def __init__(self, records: Sequence[str], *, sublinear_tf: bool = True) -> None:
        self.records = list(records)
        self.sublinear_tf = sublinear_tf
        self.vocabulary: dict[str, int] = {}
        document_frequency: Counter[str] = Counter()

        tokenized = [tokenize(record) for record in self.records]
        for tokens in tokenized:
            for token in set(tokens):
                self.vocabulary.setdefault(token, len(self.vocabulary))
                document_frequency[token] += 1

        total = len(self.records)
        self.idf = {
            token: math.log((1 + total) / (1 + count)) + 1.0
            for token, count in document_frequency.items()
        }
        self.matrix = np.vstack([self._vector(tokens) for tokens in tokenized]) if tokenized else np.zeros((0, 0))

    def _vector(self, tokens: Sequence[str]) -> np.ndarray:
        vector = np.zeros(len(self.vocabulary))
        for token, count in Counter(tokens).items():
            index = self.vocabulary.get(token)
            if index is None:
                continue
            tf = 1.0 + math.log(count) if self.sublinear_tf else float(count)
            vector[index] = tf * self.idf[token]
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def similarity(self, query: str) -> np.ndarray:
        """Cosine similarity of ``query`` against every record."""
        return self.matrix @ self._vector(tokenize(query))


class KeywordRetriever:
    """Naive RAG: rank records by the number of shared query keywords."""

    def __init__(self, records: Sequence[str]) -> None:
        self.records = list(records)

    def retrieve(self, query: str, *, top_k: int = 1) -> list[str]:
        query_keywords = keywords(query)
        scored = [
            (len(query_keywords & keywords(record)), record)
            for record in self.records
        ]
        scored = [(score, record) for score, record in scored if score > 0]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:top_k]]


class VectorRetriever:
    """Advanced RAG (§3.1): brute-force cosine, vectorizer fit per pair."""

    def __init__(self, records: Sequence[str]) -> None:
        self.records = list(records)

    def retrieve(self, query: str, *, top_k: int = 1) -> list[str]:
        scored: list[tuple[float, str]] = []
        for record in self.records:
            index = TfidfIndex([query, record])
            scored.append((float(index.matrix[0] @ index.matrix[1]), record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:top_k]]


class IndexRetriever:
    """Advanced RAG (§3.2): query a precomputed TF-IDF matrix."""

    def __init__(self, records: Sequence[str]) -> None:
        self.records = list(records)
        self.index = TfidfIndex(records)

    def retrieve(self, query: str, *, top_k: int = 1) -> list[str]:
        scores = self.index.similarity(query)
        order = np.argsort(scores)[::-1][:top_k]
        return [self.records[int(position)] for position in order]


class ModularRetriever:
    """Modular RAG: dispatch to ``keyword``, ``vector``, or ``indexed`` search."""

    METHODS = {
        "keyword": KeywordRetriever,
        "vector": VectorRetriever,
        "indexed": IndexRetriever,
    }

    def __init__(self, records: Sequence[str], method: str = "vector") -> None:
        try:
            factory = self.METHODS[method]
        except KeyError:
            raise ValueError(
                f"Unknown retrieval method {method!r}; available: {', '.join(self.METHODS)}"
            ) from None
        self.method = method
        self._retriever = factory(records)

    def retrieve(self, query: str, *, top_k: int = 1) -> list[str]:
        return self._retriever.retrieve(query, top_k=top_k)


def create_retriever(variant: str, records: Sequence[str], *, method: str = "indexed"):
    """Build the retriever for a RAG ``variant`` (chapter 1, part 2).

    ``none`` does plain generation (part 1); ``naive`` is keyword search;
    ``advanced`` is TF-IDF search (``method='vector'`` brute-force, otherwise the
    precomputed ``indexed`` matrix); ``modular`` defers to ``method``.
    """
    if variant == "none":
        return None
    if variant == "naive":
        return KeywordRetriever(records)
    if variant == "advanced":
        return VectorRetriever(records) if method == "vector" else IndexRetriever(records)
    if variant == "modular":
        return ModularRetriever(records, method=method)
    raise ValueError(f"Unknown RAG variant {variant!r}")
