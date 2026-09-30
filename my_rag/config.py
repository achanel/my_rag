"""Configuration: `.env` loading and runtime settings.

Defaults target a local Ollama server (see the root ``README.md``); the same
OpenAI-compatible settings also describe OpenAI itself, so switching providers
is only a matter of configuration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_PROVIDER = "ollama"

#: Per-provider endpoint/model defaults, overridable by env vars or CLI flags.
#: The embedding model is a free, lightweight one for Ollama (``all-minilm`` is
#: the same all-MiniLM-L6-v2 the book uses for evaluation); OpenAI keeps the
#: book's ``text-embedding-3-small``.
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:3b",
        "embedding_model": "all-minilm",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
    },
}

#: Where chapter 2 embeddings are indexed (git-ignored, like the rest of ``data/``).
DEFAULT_VECTOR_STORE = Path("data/processed/vector_store")

#: Character chunk size for the vector store, matching the book's notebook.
CHUNK_SIZE = 1000


def load_env(filename: str = ".env", start: Optional[Path] = None) -> Optional[Path]:
    """Load a `.env` file into ``os.environ``, searching upward from ``start``.

    Dependency-free stand-in for ``python-dotenv``: existing environment
    variables are never overwritten. Returns the loaded path, or ``None`` when
    no `.env` file is found.
    """
    directory = (start or Path.cwd()).resolve()
    for candidate in (directory, *directory.parents):
        env_path = candidate / filename
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
            return env_path
    return None


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration for a single run."""

    provider: str
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.1
    think: bool = False
    embedding_model: str = ""

    @classmethod
    def from_env(
        cls,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        think: Optional[bool] = None,
        embedding_model: Optional[str] = None,
    ) -> "Settings":
        """Build settings from explicit overrides, then env vars, then defaults."""
        provider = (provider or os.environ.get("MY_RAG_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
        defaults = PROVIDER_DEFAULTS.get(provider, {})

        model = model or os.environ.get("MY_RAG_MODEL") or defaults.get("model", "")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL") or defaults.get("base_url", "")
        embedding_model = (
            embedding_model
            or os.environ.get("MY_RAG_EMBEDDING_MODEL")
            or defaults.get("embedding_model", "")
        )

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key and provider == "ollama":
            api_key = "ollama"  # Ollama ignores the key, but the SDK requires a non-empty one

        if temperature is None:
            temperature = float(os.environ.get("MY_RAG_TEMPERATURE", "0.1"))

        if think is None:
            think = os.environ.get("MY_RAG_THINK", "false").strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            think=think,
            embedding_model=embedding_model,
        )
