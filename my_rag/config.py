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
        "embedding_dimension": "384",
        "vision_model": "gemma3:12b",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimension": "1536",
        "vision_model": "gpt-4o",
    },
    #: Chapter 8's book runs a local Hugging Face Llama through ``transformers``.
    #: Generation therefore ignores ``base_url``; embeddings still come from the
    #: local Ollama ``all-minilm`` endpoint (the same all-MiniLM-L6-v2 chapter 2
    #: uses), so ``base_url`` points there.
    #:
    #: The default is a **small, ungated, local** instruct model
    #: (``Qwen2.5-0.5B-Instruct``, ~1 GB, runs on CPU) — no HF token and no
    #: license gate. The book's exact ``meta-llama/Llama-2-7b-chat-hf`` is gated
    #: (accept its terms + a token); pass it with ``--model`` / ``MY_RAG_MODEL``
    #: only if you want it, and the ungated default is used if it is inaccessible.
    "hf": {
        "base_url": "http://localhost:11434/v1",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "embedding_model": "all-minilm",
        "embedding_dimension": "384",
        "vision_model": "",
    },
    #: Chapter 9's open-source local fine-tuning path: Apple's MLX + ``mlx-lm``.
    #: The base model is a small 4-bit MLX community build (Apple-only); the graph
    #: is trained locally with LoRA (``--fine-tune --provider mlx``) and run either
    #: through ``mlx_lm.generate``/``mlx_lm.server`` or ``--provider mlx``.
    #: Embeddings still come from the local Ollama ``all-minilm`` endpoint, so
    #: ``base_url`` points there (``mlx`` generation ignores it).
    "mlx": {
        "base_url": "http://localhost:11434/v1",
        "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "embedding_model": "all-minilm",
        "embedding_dimension": "384",
        "vision_model": "",
    },
}

#: Where chapter 2 embeddings are indexed (git-ignored, like the rest of ``data/``).
DEFAULT_VECTOR_STORE = Path("data/processed/vector_store")

#: Where chapter 3's LlamaIndex/Deep Lake vector index persists (git-ignored).
#: Kept apart from ``DEFAULT_VECTOR_STORE``: the LlamaIndex integration uses a
#: different dataset schema than chapter 2's raw ``deeplake.VectorStore``.
DEFAULT_INDEX_STORE = Path("data/processed/index_store")

#: Where chapter 4's drone images and their annotations are saved (git-ignored).
DEFAULT_IMAGE_DIR = Path("data/raw/drone_images")

#: Where chapter 4 writes the images with the object boxes drawn on them (git-ignored).
DEFAULT_BOXED_DIR = Path("data/processed/multimodal")

#: Where chapter 5's human-expert feedback loop saves its flashcard (git-ignored).
DEFAULT_EXPERT_FEEDBACK = Path("data/processed/adaptive/expert_feedback.txt")

#: Where chapter 6's bank-customer-churn dataset is downloaded (git-ignored).
DEFAULT_CHURN_CSV = Path("data/raw/churn/data1.csv")

#: Where chapter 6's Qdrant index persists (git-ignored). Qdrant's Python client
#: embeds an on-disk engine, so no server or API key is needed; set ``QDRANT_URL``
#: to point at a remote server instead.
DEFAULT_QDRANT_PATH = Path("data/processed/qdrant")

#: Chapter 6 Qdrant collection. The book names its Pinecone index for the 50 000
#: vectors (10 000 customer records duplicated 5 times); the same name is reused
#: here as the collection name (``MY_RAG_QDRANT_COLLECTION`` overrides it).
DEFAULT_QDRANT_COLLECTION = "bank-index-50000"

#: Where chapter 7's knowledge-graph index (triplets, nodes, embeddings) persists
#: (git-ignored). The book rebuilds its graph in memory on every run.
DEFAULT_GRAPH_STORE = Path("data/processed/graph_store")

#: Default folder for the chapter 7 pyvis HTML files (git-ignored).
DEFAULT_GRAPH_HTML_DIR = Path("data/processed/graph")

#: How many chapter 7 pages build the graph by default. The book's corpus is the
#: 100-page Marketing citation list, but extraction costs one LLM call per chunk,
#: so a run over all of it takes hours on a local model; 15 pages (~30 chunks,
#: ≈2 minutes) keeps the command runnable, ``--limit`` scales it to 100.
DEFAULT_GRAPH_LIMIT = 15

#: Where chapter 8's SciQ question/answer dataset is cached (git-ignored). Unlike
#: the book — which loads it from Hugging Face ``datasets`` in every Colab run —
#: it is fetched once and kept as JSONL, so ``--rag dynamic`` works offline.
DEFAULT_SCIENCE_QA = Path("data/raw/sciq/sciq.jsonl")

#: How many SciQ rows chapter 8 embeds by default. The book embeds all 11 679
#: supports, but every embedding here is one HTTP round-trip to Ollama
#: (``all-minilm``); 500 keeps a session runnable, ``--limit`` scales it to the
#: full dataset.
DEFAULT_SCIENCE_LIMIT = 500

#: Chapter 8's Chroma collection name (the book's ``sciq_supports6``). Chroma's
#: client runs in memory (ephemeral), so the collection is temporary by design.
DEFAULT_CHROMA_COLLECTION = "sciq_supports6"

#: Where chapter 9's chat fine-tuning file is written (git-ignored). The book
#: writes ``QA_prompts_and_completions.json`` on Drive; here it is a JSONL of
#: ``{"messages": [...]}`` items built from the same SciQ dataset chapter 8
#: caches, so it needs no network once SciQ is present. It feeds the local MLX
#: training directly (``train.jsonl`` is split out of it).
DEFAULT_FINETUNE_DATA = Path("data/processed/finetune/qa_prompts_and_completions.jsonl")

#: How many SciQ rows chapter 9 turns into fine-tuning examples by default. The
#: book uses all 10 481 filtered rows; a smaller default keeps a local training
#: run quick while ``--limit`` scales it up.
DEFAULT_FINETUNE_LIMIT = 500

#: Where chapter 9's local MLX dataset (``train.jsonl`` etc.) is written
#: (git-ignored). ``mlx_lm.lora --data`` expects a directory with ``train.jsonl``.
DEFAULT_MLX_DATA = Path("data/processed/finetune/mlx")

#: Where chapter 9's local MLX LoRA adapters are saved / loaded (git-ignored).
DEFAULT_MLX_ADAPTERS = Path("data/processed/finetune/adapters")

#: How many MLX LoRA training iterations to run by default (``--iters``).
DEFAULT_MLX_ITERS = 600

#: Where the Hugging Face access token is read from. The book keeps its token in
#: a file and exports it as ``HF_TOKEN``; the default path is this machine's
#: ``~/.config/hugging_face/token`` (override with ``MY_RAG_HF_TOKEN_FILE``, or
#: set ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` directly).
DEFAULT_HF_TOKEN_FILE = Path.home() / ".config" / "hugging_face" / "token"

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


def load_hf_token(
    filename: Optional[Path] = None,
    *,
    env_var: str = "HF_TOKEN",
) -> Optional[str]:
    """Export the Hugging Face token from a file into ``os.environ``.

    Mirrors the book's secret step (read the token file, set ``HF_TOKEN``), which
    ``transformers`` / ``huggingface_hub`` then use to reach gated models such as
    ``meta-llama/Llama-2-7b-chat-hf``. An existing ``HF_TOKEN`` /
    ``HUGGING_FACE_HUB_TOKEN`` wins (so Colab secrets and CI keep working), and
    ``MY_RAG_HF_TOKEN_FILE`` overrides the path. Returns the token, or ``None``
    when no file (or an empty one) is found. The value is never logged.
    """
    existing = os.environ.get(env_var) or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if existing:
        return existing
    path = Path(os.environ.get("MY_RAG_HF_TOKEN_FILE") or filename or DEFAULT_HF_TOKEN_FILE)
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    token = content.strip().split("\n", 1)[0].strip() if content.strip() else ""
    if not token:
        return None
    os.environ[env_var] = token
    return token


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
    embedding_dimension: int = 0
    vision_model: str = ""
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: Path = DEFAULT_QDRANT_PATH
    qdrant_collection: str = DEFAULT_QDRANT_COLLECTION
    graph_store: Path = DEFAULT_GRAPH_STORE
    graph_limit: int = DEFAULT_GRAPH_LIMIT
    science_qa: Path = DEFAULT_SCIENCE_QA
    science_limit: int = DEFAULT_SCIENCE_LIMIT
    chroma_collection: str = DEFAULT_CHROMA_COLLECTION
    finetune_data: Path = DEFAULT_FINETUNE_DATA
    finetune_limit: int = DEFAULT_FINETUNE_LIMIT
    mlx_data: Path = DEFAULT_MLX_DATA
    mlx_adapters: Path = DEFAULT_MLX_ADAPTERS
    mlx_iters: int = DEFAULT_MLX_ITERS

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
        vision_model: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        qdrant_collection: Optional[str] = None,
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
        vision_model = (
            vision_model
            or os.environ.get("MY_RAG_VISION_MODEL")
            or defaults.get("vision_model", "")
        )

        if embedding_dimension is None:
            override = os.environ.get("MY_RAG_EMBEDDING_DIMENSION")
            embedding_dimension = int(override or defaults.get("embedding_dimension", "0") or 0)

        qdrant_collection = (
            qdrant_collection
            or os.environ.get("MY_RAG_QDRANT_COLLECTION")
            or DEFAULT_QDRANT_COLLECTION
        )

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key and provider in {"ollama", "hf", "mlx"}:
            # Ollama ignores the key; the ``hf``/``mlx`` providers generate locally
            # but their embeddings still go to Ollama, so both need a non-empty SDK key.
            api_key = "ollama"

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
            embedding_dimension=embedding_dimension,
            vision_model=vision_model,
            qdrant_url=os.environ.get("QDRANT_URL", "").strip(),
            qdrant_api_key=os.environ.get("QDRANT_API_KEY", "").strip(),
            qdrant_path=Path(os.environ.get("MY_RAG_QDRANT_PATH") or DEFAULT_QDRANT_PATH),
            qdrant_collection=qdrant_collection,
            graph_store=Path(os.environ.get("MY_RAG_GRAPH_STORE") or DEFAULT_GRAPH_STORE),
            graph_limit=int(os.environ.get("MY_RAG_GRAPH_LIMIT") or DEFAULT_GRAPH_LIMIT),
            science_qa=Path(os.environ.get("MY_RAG_SCIENCE_QA") or DEFAULT_SCIENCE_QA),
            science_limit=int(os.environ.get("MY_RAG_SCIENCE_LIMIT") or DEFAULT_SCIENCE_LIMIT),
            chroma_collection=(
                os.environ.get("MY_RAG_CHROMA_COLLECTION") or DEFAULT_CHROMA_COLLECTION
            ),
            finetune_data=Path(os.environ.get("MY_RAG_FINETUNE_DATA") or DEFAULT_FINETUNE_DATA),
            finetune_limit=int(
                os.environ.get("MY_RAG_FINETUNE_LIMIT") or DEFAULT_FINETUNE_LIMIT
            ),
            mlx_data=Path(os.environ.get("MY_RAG_MLX_DATA") or DEFAULT_MLX_DATA),
            mlx_adapters=Path(os.environ.get("MY_RAG_MLX_ADAPTERS") or DEFAULT_MLX_ADAPTERS),
            mlx_iters=int(os.environ.get("MY_RAG_MLX_ITERS") or DEFAULT_MLX_ITERS),
        )
