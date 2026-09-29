"""The retrieve -> augment -> generate pipeline.

Chapter 1, part 1 ("Foundations and Basic Implementation") only uses the
generator ("generation without augmentation"), so the retriever is optional.
Later parts plug a concrete :class:`Retriever` in without touching the
generator: keyword search, vector search, index-based search, etc.
"""
from __future__ import annotations

from typing import Iterator, Optional, Protocol, runtime_checkable

from .llm import LLM

DEFAULT_SYSTEM_PROMPT = "You are an expert Natural Language Processing exercise expert."
DEFAULT_INSTRUCTION = "Please elaborate on the following content:"


@runtime_checkable
class Retriever(Protocol):
    """Extension point for retrieval strategies (chapter 1, part 2 onward)."""

    def retrieve(self, query: str, *, top_k: int = 1) -> list[str]:
        """Return the ``top_k`` records most relevant to ``query``."""


class RAGPipeline:
    """Generate an answer for a query, optionally augmented with retrieved context."""

    def __init__(
        self,
        llm: LLM,
        retriever: Optional[Retriever] = None,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.instruction = instruction

    def augment(self, query: str) -> str:
        """Combine the query with retrieved context (no-op without a retriever)."""
        if self.retriever is None:
            return query
        context = "\n".join(self.retriever.retrieve(query))
        return f"{query}: {context}" if context else query

    def build_prompt(self, query: str) -> str:
        """Assemble the final prompt from the instruction and augmented query."""
        return f"{self.instruction}\n{self.augment(query)}"

    def answer(self, query: str) -> str:
        """Run the full pipeline and return the model's response."""
        return self.llm.complete(self.build_prompt(query), system=self.system_prompt)

    def stream(self, query: str) -> Iterator[str]:
        """Run the pipeline and yield the response token by token."""
        return self.llm.stream(self.build_prompt(query), system=self.system_prompt)
