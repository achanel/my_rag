"""my_rag — a small, extensible RAG playground.

Built while working through *RAG-Driven Generative AI* (Denis Rothman, Packt).
Chapter 1, part 1 ("Foundations and Basic Implementation") is implemented as
:class:`~my_rag.pipeline.RAGPipeline` over a pluggable
:class:`~my_rag.llm.LLM` provider, defaulting to a local Ollama server. Part 2
("Advanced Techniques and Evaluation") adds the Naive, Advanced, and Modular
RAG retrievers in :mod:`my_rag.retrieval`. Chapter 2 adds data collection
(:mod:`my_rag.collection`), embeddings (:mod:`my_rag.embeddings`), and the Deep
Lake vector store (:mod:`my_rag.vectorstore`).

The vector-store symbols are resolved lazily (PEP 562) so that ``import my_rag``
does not pull in Deep Lake unless it is actually used.
"""
from .collection import Article, CollectionResult, WIKI_URLS, clean_text, collect, extract_text
from .config import CHUNK_SIZE, DEFAULT_VECTOR_STORE, Settings, load_env
from .embeddings import EmbeddingFunction, OpenAICompatibleEmbedding, create_embedding_function
from .llm import LLM, create_llm, register_provider
from .pipeline import RAGPipeline, Retriever
from .retrieval import (
    IndexRetriever,
    KeywordRetriever,
    ModularRetriever,
    TfidfIndex,
    VectorRetriever,
    create_retriever,
)

__all__ = [
    "Article",
    "CollectionResult",
    "WIKI_URLS",
    "clean_text",
    "collect",
    "extract_text",
    "Settings",
    "load_env",
    "CHUNK_SIZE",
    "DEFAULT_VECTOR_STORE",
    "LLM",
    "create_llm",
    "register_provider",
    "EmbeddingFunction",
    "OpenAICompatibleEmbedding",
    "create_embedding_function",
    "RAGPipeline",
    "Retriever",
    "TfidfIndex",
    "KeywordRetriever",
    "VectorRetriever",
    "IndexRetriever",
    "ModularRetriever",
    "create_retriever",
    "DeepLakeVectorStore",
    "BuildResult",
    "SearchHit",
    "chunk_text",
    "open_vector_store",
]

#: Vector-store names resolved on first access (holds back the Deep Lake import).
_VECTOR_STORE_EXPORTS = frozenset(
    {"DeepLakeVectorStore", "BuildResult", "SearchHit", "chunk_text", "open_vector_store"}
)


def __getattr__(name: str):
    if name in _VECTOR_STORE_EXPORTS:
        from . import vectorstore

        return getattr(vectorstore, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
