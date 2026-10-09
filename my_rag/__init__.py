"""my_rag — a small, extensible RAG playground.

Built while working through *RAG-Driven Generative AI* (Denis Rothman, Packt).
Chapter 1, part 1 ("Foundations and Basic Implementation") is implemented as
:class:`~my_rag.pipeline.RAGPipeline` over a pluggable
:class:`~my_rag.llm.LLM` provider, defaulting to a local Ollama server. Part 2
("Advanced Techniques and Evaluation") adds the Naive, Advanced, and Modular
RAG retrievers in :mod:`my_rag.retrieval`. Chapter 2 adds data collection
(:mod:`my_rag.collection`), embeddings (:mod:`my_rag.embeddings`), and the Deep
Lake vector store (:mod:`my_rag.vectorstore`). Chapter 3 adds LlamaIndex
index-based semantic search (:mod:`my_rag.indexing`). Chapter 4 adds multimodal,
modular RAG over text and drone images (:mod:`my_rag.multimodal`). Chapter 5 adds
ranking-adaptive RAG with a human-feedback loop (:mod:`my_rag.adaptive`). Chapter 6
adds the bank-churn dataset, a NumPy KMeans segmentation and a Qdrant-backed index
(:mod:`my_rag.scaling`; imported directly, like the other optional modules).

The vector-store and index symbols are resolved lazily (PEP 562) so that
``import my_rag`` does not pull in Deep Lake or LlamaIndex unless actually used.
"""
from .adaptive import (
    ADAPTIVE_SOURCES,
    HUMAN_FEEDBACK,
    AdaptiveRAG,
    AdaptiveResult,
    Evaluation,
    NoMatchError,
    RatingState,
    Retrieved,
    evaluate,
    first_words,
    load_expert_feedback,
    match_keyword,
    retrieve,
    save_expert_feedback,
    strategy_for_ranking,
    tfidf_similarity,
)
from .collection import (
    Article,
    CollectionResult,
    DEFAULT_DRONE_OUTPUT,
    DRONE_URLS,
    WIKI_URLS,
    clean_text,
    collect,
    extract_text,
)
from .config import (
    CHUNK_SIZE,
    DEFAULT_EXPERT_FEEDBACK,
    DEFAULT_IMAGE_DIR,
    DEFAULT_INDEX_STORE,
    DEFAULT_VECTOR_STORE,
    Settings,
    load_env,
)
from .embeddings import (
    EmbeddingFunction,
    EmbeddingIndex,
    OpenAICompatibleEmbedding,
    cosine_similarity,
    create_embedding_function,
)
from .llm import LLM, create_llm, register_provider
from .multimodal import (
    MultimodalAnswer,
    MultimodalRAG,
    Sample,
    collect_visdrone,
    load_samples,
    load_text_chunks,
)
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
    "DRONE_URLS",
    "DEFAULT_DRONE_OUTPUT",
    "clean_text",
    "collect",
    "extract_text",
    "ADAPTIVE_SOURCES",
    "HUMAN_FEEDBACK",
    "AdaptiveRAG",
    "AdaptiveResult",
    "Evaluation",
    "NoMatchError",
    "RatingState",
    "Retrieved",
    "evaluate",
    "first_words",
    "load_expert_feedback",
    "match_keyword",
    "retrieve",
    "save_expert_feedback",
    "strategy_for_ranking",
    "tfidf_similarity",
    "Settings",
    "load_env",
    "CHUNK_SIZE",
    "DEFAULT_VECTOR_STORE",
    "DEFAULT_INDEX_STORE",
    "DEFAULT_IMAGE_DIR",
    "DEFAULT_EXPERT_FEEDBACK",
    "LLM",
    "create_llm",
    "register_provider",
    "EmbeddingFunction",
    "EmbeddingIndex",
    "OpenAICompatibleEmbedding",
    "cosine_similarity",
    "create_embedding_function",
    "MultimodalAnswer",
    "MultimodalRAG",
    "Sample",
    "collect_visdrone",
    "load_samples",
    "load_text_chunks",
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
    "IndexQueryEngine",
    "IndexAnswer",
    "Source",
    "INDEX_KINDS",
    "DEFAULT_TOP_K",
    "build_index",
    "build_llm",
    "build_embed_model",
    "deep_lake_index_store_class",
    "load_documents",
]

#: Vector-store names resolved on first access (holds back the Deep Lake import).
_VECTOR_STORE_EXPORTS = frozenset(
    {"DeepLakeVectorStore", "BuildResult", "SearchHit", "chunk_text", "open_vector_store"}
)

#: LlamaIndex names resolved on first access (holds back the LlamaIndex import).
_INDEX_EXPORTS = frozenset(
    {
        "IndexQueryEngine",
        "IndexAnswer",
        "Source",
        "INDEX_KINDS",
        "DEFAULT_TOP_K",
        "build_index",
        "build_llm",
        "build_embed_model",
        "deep_lake_index_store_class",
        "load_documents",
    }
)


def __getattr__(name: str):
    if name in _VECTOR_STORE_EXPORTS:
        from . import vectorstore

        return getattr(vectorstore, name)
    if name in _INDEX_EXPORTS:
        from . import indexing

        return getattr(indexing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
