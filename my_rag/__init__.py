"""my_rag — a small, extensible RAG playground.

Built while working through *RAG-Driven Generative AI* (Denis Rothman, Packt).
Chapter 1, part 1 ("Foundations and Basic Implementation") is implemented as
:class:`~my_rag.pipeline.RAGPipeline` over a pluggable
:class:`~my_rag.llm.LLM` provider, defaulting to a local Ollama server. Part 2
("Advanced Techniques and Evaluation") adds the Naive, Advanced, and Modular
RAG retrievers in :mod:`my_rag.retrieval`.
"""
from .config import Settings, load_env
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
    "Settings",
    "load_env",
    "LLM",
    "create_llm",
    "register_provider",
    "RAGPipeline",
    "Retriever",
    "TfidfIndex",
    "KeywordRetriever",
    "VectorRetriever",
    "IndexRetriever",
    "ModularRetriever",
    "create_retriever",
]
