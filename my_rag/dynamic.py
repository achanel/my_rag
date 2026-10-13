"""Dynamic RAG with Chroma and a Hugging Face Llama (chapter 8).

Reimplementation of ``Chapter08/Dynamic_RAG_with_Chroma_and_Hugging_Face.ipynb``:
a *dynamic* (temporary, per-session) RAG over the SciQ science-QA dataset.

The book's scenario is a daily meeting: load SciQ, build the ``completion``
(``correct_answer + " because " + support``), embed the completions into a
throw-away Chroma collection, query it, score the retrieval with a semantic
cosine, generate a beginner summary with a local Hugging Face Llama, then delete
the collection. The pipeline:

- **Prepare** — :func:`fetch_science_qa` walks the Hugging Face datasets server
  for ``allenai/sciq``, keeps rows with a support and a correct answer, drops the
  three distractor columns, and builds the ``completion``; :func:`collect_science_qa`
  caches the result as JSONL so later runs need no network.
- **Index** — :func:`open_science_collection` opens the book's ephemeral
  (in-memory) Chroma collection and :func:`index_rows` embeds the completions
  through the configured embedding model (the book relies on Chroma's default
  ``all-MiniLM-L6-v2``).
- **Query / evaluate** — :meth:`ChromaIndex.query` returns the nearest completion;
  :meth:`DynamicRAG.evaluate` reproduces the notebook's retrieval loop and its
  ``similarity > 0.7`` accuracy.
- **Generate** — :meth:`DynamicRAG.answer` concatenates the book's instruction with
  the retrieved support and asks the model for a summary.

Substitutions, consistent with the rest of ``my_rag``:

- **Local Ollama by default instead of the Hugging Face Llama pipeline.** The
  book loads ``meta-llama/Llama-2-7b-chat-hf`` in ``transformers`` (a 7B model,
  a GPU and a HF token). The default here is the same local ``qwen2.5:3b`` the
  other chapters use; ``--provider hf`` selects :class:`my_rag.llm.HuggingFaceLLM`,
  which runs the book's ``transformers.pipeline`` faithfully when you want it.
- **Ollama ``all-minilm`` embeddings instead of Chroma's default embedding.**
  The book lets Chroma pick ``all-MiniLM-L6-v2`` (downloaded as ONNX on first
  use); here the embedding function is chapter 2's ``all-minilm`` (the same
  all-MiniLM-L6-v2 family, served by Ollama), so one embedding stack covers the
  whole repo. It is injected into the collection, so a test can pass a fake.
- **The Hugging Face datasets server over ``requests`` instead of the ``datasets``
  library.** The book calls ``load_dataset("sciq")`` (pulls ``pyarrow``); here the
  same rows come from the public datasets-server REST API and are cached to
  ``data/raw/sciq/sciq.jsonl``, so the dataset is fetched once and offline after.
- **A semantic embedding cosine instead of spaCy ``en_core_web_md``.** The book's
  ``simple_text_similarity`` loads a spaCy model to score retrieval against the
  known completion; here :func:`text_similarities` reuses the configured embedding
  model and chapter 2's :func:`~my_rag.embeddings.cosine_similarity`.

Not ported: the Colab scaffolding (the Drive mount, ``hf_token.txt`` / HF secret,
the ``!pip install`` and ``spacy download`` cells), the notebook's inline
``display()``/``print`` cells, and the ``collection.get()`` embedding dump (a
display step, not part of the pipeline).

Chroma's client is **ephemeral**: the collection lives in memory for one run and
is deleted at the end, which is the notebook's "lightweight temporary vector
storage" — there is no persistent Chroma directory to clean up.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import requests

from .config import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_SCIENCE_LIMIT,
    DEFAULT_SCIENCE_QA,
    Settings,
)
from .embeddings import EmbeddingFunction, cosine_similarity, create_embedding_function
from .llm import LLM, create_llm

#: SciQ on the Hugging Face datasets hub and the public datasets server that
#: serves it as JSON rows (no ``datasets``/``pyarrow`` dependency needed).
SCIENCE_QA_DATASET = "allenai/sciq"
SCIENCE_QA_CONFIG = "default"
SCIENCE_QA_SPLIT = "train"
SCIENCE_QA_ROWS_URL = "https://datasets-server.huggingface.co/rows"

#: Rows requested per datasets-server call (its default page size).
SCIENCE_QA_PAGE = 100

#: Attempts per page (and the base back-off, seconds) for transient 5xx answers.
SCIENCE_QA_RETRIES = 4
SCIENCE_QA_RETRY_DELAY = 2.0

#: The columns the book drops right after loading SciQ.
DROP_COLUMNS: tuple[str, ...] = ("distractor3", "distractor1", "distractor2")

#: The fields kept in the cached JSONL (SciQ minus the distractors, plus the completion).
SCIENCE_QA_FIELDS: tuple[str, ...] = ("question", "correct_answer", "support", "completion")

#: The book's ``n_results`` for a query and its accuracy threshold (``> 0.7``).
DEFAULT_TOP_K = 1
SIMILARITY_THRESHOLD = 0.7

#: The book's initial prompt and its summary instruction (``iprompt``).
DEFAULT_QUESTION = "Millions of years ago, plants used energy from the sun to form what?"
SUMMARY_INSTRUCTION = "Read the following input and write a summary for beginners."

#: The book's generation temperature for ``LLaMA2`` (``do_sample`` at 0.5).
GENERATION_TEMPERATURE = 0.5

_HEADERS = {"User-Agent": "my_rag/0.1 (study project; RAG-Driven-Generative-AI chapter 8)"}


# --------------------------------------------------------------------------- #
# Prepare — the SciQ dataset
# --------------------------------------------------------------------------- #
def prepare_row(row: dict) -> dict:
    """Keep the question, answer and support; drop the distractors; add the completion.

    The book's cell: ``drop(columns=['distractor3', 'distractor1', 'distractor2'])``
    then ``completion = correct_answer + " because " + support``.
    """
    question = str(row.get("question", "")).strip()
    correct_answer = str(row.get("correct_answer", "")).strip()
    support = str(row.get("support", "")).strip()
    return {
        "question": question,
        "correct_answer": correct_answer,
        "support": support,
        "completion": f"{correct_answer} because {support}",
    }


def prepare_rows(raw_rows: Sequence[dict]) -> list[dict]:
    """Filter and prepare SciQ rows, as the book's ``filter`` + ``dropna`` cells.

    Rows without a support or a correct answer are dropped (the notebook keeps
    only ``support != '' and correct_answer != ''``).
    """
    prepared = []
    for row in raw_rows:
        if not str(row.get("support", "")).strip() or not str(row.get("correct_answer", "")).strip():
            continue
        prepared.append(prepare_row(row))
    return prepared


def fetch_science_qa(
    limit: int = DEFAULT_SCIENCE_LIMIT,
    *,
    session: Optional[requests.Session] = None,
    page_size: int = SCIENCE_QA_PAGE,
    timeout: float = 30.0,
    progress: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Page the datasets server for SciQ and return the first ``limit`` valid rows.

    ``limit`` counts *kept* rows, so pages whose rows are filtered out (SciQ has
    rows with an empty support) do not shrink the result.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    if page_size < 1:
        raise ValueError(f"page_size must be at least 1, got {page_size}")
    client = session or requests
    kept: list[dict] = []
    offset = 0
    total: Optional[int] = None
    while len(kept) < limit:
        for attempt in range(SCIENCE_QA_RETRIES):
            response = client.get(
                SCIENCE_QA_ROWS_URL,
                params={
                    "dataset": SCIENCE_QA_DATASET,
                    "config": SCIENCE_QA_CONFIG,
                    "split": SCIENCE_QA_SPLIT,
                    "offset": offset,
                    "length": page_size,
                },
                headers=_HEADERS,
                timeout=timeout,
            )
            # The datasets server answers 502/503 now and then; retry those pages.
            if response.status_code < 500 or attempt == SCIENCE_QA_RETRIES - 1:
                break
            time.sleep(SCIENCE_QA_RETRY_DELAY * (attempt + 1))
        response.raise_for_status()
        payload = response.json()
        total = payload.get("num_rows_total", total or 0)
        rows = [item["row"] for item in payload.get("rows", [])]
        if not rows:
            break
        kept.extend(prepare_rows(rows))
        offset += len(rows)
        if progress is not None:
            progress(f"fetched {offset}/{total} rows, kept {len(kept)}")
        if total and offset >= total:
            break
    return kept[:limit]


def collect_science_qa(
    path: Path | str = DEFAULT_SCIENCE_QA,
    *,
    limit: int = DEFAULT_SCIENCE_LIMIT,
    session: Optional[requests.Session] = None,
    page_size: int = SCIENCE_QA_PAGE,
    timeout: float = 30.0,
    progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """Download ``limit`` SciQ rows and cache them to ``path`` as JSONL."""
    path = Path(path)
    rows = fetch_science_qa(
        limit, session=session, page_size=page_size, timeout=timeout, progress=progress
    )
    if not rows:
        raise ValueError(f"{SCIENCE_QA_DATASET} returned no usable rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_science_qa(path: Path | str = DEFAULT_SCIENCE_QA) -> list[dict]:
    """Read the cached SciQ JSONL into a list of row dicts."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"SciQ dataset {path} not found; collect it with: "
            "python -m my_rag --collect --corpus sciq"
        )
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"SciQ dataset {path} is empty")
    return rows


# --------------------------------------------------------------------------- #
# Index — the Chroma collection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QAMatch:
    """One Chroma hit: its id, distance and the stored completion text."""

    id: str
    distance: float
    text: str

    def __str__(self) -> str:  # so --show-context prints the completion
        return self.text


class ChromaEmbedding:
    """Adapt :class:`~my_rag.embeddings.EmbeddingFunction` to Chroma's protocol.

    Chroma calls an embedding function through ``__call__`` / ``embed_documents``
    (documents) and ``embed_query`` (queries); ``name`` and the config hooks let a
    collection serialise it. Both directions delegate to the same
    ``embed_documents`` of the wrapped function.

    The class is registered with Chroma (see :func:`_register_chroma_embedding`)
    so a collection can rebuild it from config; the rebuilt instance carries no
    function (the live collection keeps the injected one).
    """

    def __init__(self, embedding_function: Optional[EmbeddingFunction] = None) -> None:
        self._function = embedding_function

    def _embed(self, input: Sequence[str]) -> list[list[float]]:
        if self._function is None:
            raise RuntimeError(
                "ChromaEmbedding has no backing embedding function; build it with "
                "open_science_collection() and its configured embedding model"
            )
        return self._function.embed_documents(list(input))

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        return self._embed(input)

    def embed_documents(self, input: Sequence[str]) -> list[list[float]]:
        return self._embed(input)

    def embed_query(self, input: Sequence[str]) -> list[list[float]]:
        return self._embed(input)

    @staticmethod
    def name() -> str:
        return "my_rag"

    def is_legacy(self) -> bool:
        # Tell Chroma this is a current-style function with config hooks, so it does
        # not warn about a "legacy embedding function config".
        return False

    def get_config(self) -> dict:
        return {}

    def default_space(self) -> str:
        return "l2"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def build_from_config(config: dict) -> "ChromaEmbedding":
        # Chroma rebuilds this from the stored config to read the collection's space;
        # the live collection keeps the embedding function passed to it instead.
        return ChromaEmbedding(None)


_CHROMA_REGISTERED = False


def _register_chroma_embedding() -> None:
    """Register :class:`ChromaEmbedding` with Chroma once, so config round-trips."""
    global _CHROMA_REGISTERED
    if not _CHROMA_REGISTERED:
        from chromadb.utils.embedding_functions import register_embedding_function

        register_embedding_function(ChromaEmbedding)
        _CHROMA_REGISTERED = True


class ChromaIndex:
    """A thin wrapper over one ephemeral Chroma collection (the pipeline's index)."""

    def __init__(self, client, collection, name: str) -> None:
        self.client = client
        self.collection = collection
        self.name = name

    def add(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        *,
        metadatas: Optional[Sequence[dict]] = None,
    ) -> None:
        """Add documents to the collection (Chroma embeds them with our function)."""
        self.collection.add(
            ids=list(ids),
            documents=list(documents),
            metadatas=list(metadatas) if metadatas is not None else None,
        )

    def add_rows(self, rows: Sequence[dict]) -> int:
        """Store one document per SciQ row (its ``completion``); return the count."""
        if not rows:
            raise ValueError("cannot index an empty dataset")
        self.add(
            ids=[str(index) for index in range(len(rows))],
            documents=[row["completion"] for row in rows],
            metadatas=[{"type": "completion"} for _ in rows],
        )
        return len(rows)

    def query(self, question: str, *, top_k: int = DEFAULT_TOP_K) -> list[QAMatch]:
        """The ``top_k`` nearest completions for ``question``."""
        result = self.collection.query(
            query_texts=[question], n_results=top_k, include=["documents", "distances"]
        )
        documents = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        return [
            QAMatch(id=str(ids[position]), distance=float(distances[position]), text=str(document))
            for position, document in enumerate(documents)
        ]

    def count(self) -> int:
        """Number of documents currently stored in the collection."""
        return self.collection.count()

    def delete(self) -> None:
        """Delete the collection (the notebook's ``client.delete_collection``)."""
        self.client.delete_collection(self.name)


def open_science_collection(
    settings: Settings,
    *,
    collection: Optional[str] = None,
    embedding_function: Optional[EmbeddingFunction] = None,
    client=None,
    create: bool = True,
) -> ChromaIndex:
    """Open the temporary Chroma collection, creating it when absent.

    Chroma's client runs in memory (``chromadb.Client()``), so the collection is
    ephemeral — the book's "lightweight temporary vector storage". The embedding
    function defaults to chapter 2's ``create_embedding_function``; pass one to
    inject a fake in tests. Both the client and Chroma itself are imported lazily.
    """
    if client is None:
        import chromadb

        client = chromadb.Client()
    _register_chroma_embedding()
    name = collection or settings.chroma_collection
    embedding = ChromaEmbedding(embedding_function or create_embedding_function(settings))
    if create:
        stored = client.get_or_create_collection(name=name, embedding_function=embedding)
    else:
        try:
            stored = client.get_collection(name=name)
        except Exception as exc:  # missing collection (Chroma raises a generic error)
            raise FileNotFoundError(
                f"Chroma collection {name!r} not found; build it with: python -m my_rag --rag dynamic"
            ) from exc
    return ChromaIndex(client, stored, name)


# --------------------------------------------------------------------------- #
# Similarity — the book's "simple_text_similarity"
# --------------------------------------------------------------------------- #
def text_similarities(
    first: Sequence[str], second: Sequence[str], embedding_function: EmbeddingFunction
) -> list[float]:
    """Cosine similarity of each ``first``/``second`` pair, through the embeddings.

    The book's ``simple_text_similarity`` in the evaluation loop, with the
    configured embedding model in place of spaCy's ``en_core_web_md``. The pair's
    texts are embedded in one batch.
    """
    if len(first) != len(second):
        raise ValueError(f"got {len(first)} and {len(second)} texts")
    if not first:
        return []
    vectors = embedding_function.embed_documents([*first, *second])
    split = len(first)
    return [
        cosine_similarity(vectors[position], vectors[split + position])
        for position in range(split)
    ]


def text_similarity(first: str, second: str, embedding_function: EmbeddingFunction) -> float:
    """Cosine similarity of two texts (the book's per-pair `simple_text_similarity`)."""
    return text_similarities([first], [second], embedding_function)[0]


# --------------------------------------------------------------------------- #
# Generate — prompt + retrieval
# --------------------------------------------------------------------------- #
def build_summary_prompt(context: str) -> str:
    """The book's ``lprompt``: the instruction plus the retrieved support."""
    return f"{SUMMARY_INSTRUCTION} {context}".strip()


@dataclass(frozen=True)
class DynamicAnswer:
    """Outcome of :meth:`DynamicRAG.answer`: the retrieval, prompt and summary."""

    question: str
    matches: list[QAMatch]
    retrieved: str
    prompt: str
    response: str
    elapsed: float


@dataclass(frozen=True)
class RetrievalItem:
    """One row of :meth:`DynamicRAG.evaluate`: question, retrieval, original, similarity."""

    position: int
    question: str
    retrieved: str
    original: str
    similarity: float


@dataclass(frozen=True)
class RetrievalReport:
    """The notebook's retrieval-accuracy loop: per-item scores and the overall rate."""

    documents: int
    threshold: float
    accuracy: float
    items: list[RetrievalItem] = field(default_factory=list)

    @property
    def correct(self) -> int:
        """How many retrievals scored above the threshold."""
        return sum(1 for item in self.items if item.similarity > self.threshold)


class DynamicRAG:
    """The chapter 8 pipeline over an ephemeral Chroma index."""

    def __init__(
        self,
        settings: Settings,
        index: ChromaIndex,
        *,
        embedding_function: Optional[EmbeddingFunction] = None,
        llm: Optional[LLM] = None,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self.settings = settings
        self.index = index
        self.embedding_function = embedding_function or create_embedding_function(settings)
        self.llm = llm or create_llm(settings)
        self.top_k = top_k
        self.threshold = threshold

    def retrieve(self, question: str) -> list[QAMatch]:
        """Query the collection for the nearest completions."""
        return self.index.query(question, top_k=self.top_k)

    def answer(self, question: str = DEFAULT_QUESTION) -> DynamicAnswer:
        """Retrieve the nearest support, augment the prompt, and summarize it."""
        matches = self.retrieve(question)
        retrieved = matches[0].text if matches else ""
        prompt = build_summary_prompt(retrieved)
        start = time.perf_counter()
        response = self.llm.complete(prompt, temperature=GENERATION_TEMPERATURE)
        elapsed = time.perf_counter() - start
        return DynamicAnswer(
            question=question,
            matches=matches,
            retrieved=retrieved,
            prompt=prompt,
            response=response,
            elapsed=elapsed,
        )

    def evaluate(
        self,
        rows: Sequence[dict],
        *,
        limit: Optional[int] = None,
        progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> RetrievalReport:
        """Reproduce the notebook's retrieval loop: query by question, score vs. completion.

        For the first ``limit`` rows, the question is embedded and the nearest
        stored completion retrieved; its semantic similarity to that row's known
        completion is measured, and the share above the threshold is the accuracy.
        """
        sample = list(rows)[:limit] if limit else list(rows)
        if not sample:
            raise ValueError("cannot evaluate an empty dataset")
        positions: list[tuple[int, str, str, str]] = []
        for position, row in enumerate(sample, start=1):
            matches = self.retrieve(row["question"])
            retrieved = matches[0].text if matches else ""
            positions.append((position, row["question"], retrieved, row["completion"]))
            if progress is not None:
                progress(position, len(sample), row["question"])
        similarities = text_similarities(
            [item[3] for item in positions], [item[2] for item in positions], self.embedding_function
        )
        items = [
            RetrievalItem(
                position=position,
                question=question,
                retrieved=retrieved,
                original=original,
                similarity=similarity,
            )
            for (position, question, retrieved, original), similarity in zip(positions, similarities)
        ]
        accuracy = sum(1 for item in items if item.similarity > self.threshold) / len(items)
        return RetrievalReport(
            documents=len(items), threshold=self.threshold, accuracy=accuracy, items=items
        )


__all__ = [
    "SCIENCE_QA_DATASET",
    "SCIENCE_QA_CONFIG",
    "SCIENCE_QA_SPLIT",
    "SCIENCE_QA_ROWS_URL",
    "SCIENCE_QA_PAGE",
    "SCIENCE_QA_FIELDS",
    "DROP_COLUMNS",
    "DEFAULT_CHROMA_COLLECTION",
    "DEFAULT_TOP_K",
    "SIMILARITY_THRESHOLD",
    "DEFAULT_QUESTION",
    "SUMMARY_INSTRUCTION",
    "GENERATION_TEMPERATURE",
    "prepare_row",
    "prepare_rows",
    "fetch_science_qa",
    "collect_science_qa",
    "read_science_qa",
    "QAMatch",
    "ChromaEmbedding",
    "ChromaIndex",
    "open_science_collection",
    "text_similarity",
    "text_similarities",
    "build_summary_prompt",
    "DynamicAnswer",
    "RetrievalItem",
    "RetrievalReport",
    "DynamicRAG",
]
